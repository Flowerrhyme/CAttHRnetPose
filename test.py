import numpy as np
import torch
from torch import nn
from torch.nn import init
import yaml
from modules.model import AttHRnetPose
import common.loss as Loss
from common.general import dist2bbox, make_anchors, bbox2dist, xywh2xyxy, scale_coords
import time
import torchvision
import argparse
import os 
import yaml
import torch.distributed as dist
import torch
from modules.model import AttHRnetPose
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from common.dataset import LoadImagesAndLabels, LoadImages
from tqdm import tqdm
from torch.cuda import amp
from common.loss import ComputeLoss
from pathlib import Path
import cv2
from common.myeval import Eval
from torch.utils.data import random_split

SEED = 13

IMG_H = 1
IMG_W = 1


def plot_skeleton_kpts(im, kpts, steps=3):
    if isinstance(im, torch.Tensor):
        im = im.cpu().float().numpy()
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord*IMG_W), int(y_coord*IMG_H)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]*IMG_W), int(kpts[(sk[0]-1)*steps+1]*IMG_H))
        pos2 = (int(kpts[(sk[1]-1)*steps]*IMG_W), int(kpts[(sk[1]-1)*steps+1]*IMG_H))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    return im


def bbox_decode_xywh(anchor_points, pred_dist, use_dfl=True):
    device = pred_dist.device
    if use_dfl:
        proj = torch.arange(16).float().to(device)
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
    return dist2bbox(pred_dist, anchor_points, xywh=True)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


#prediction 
def non_max_suppression(prediction, conf_thres=0.05, iou_thres=0.45,  agnostic=False):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 56 # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence


        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
            
        kpts = x[:, 6:]
        conf, j = x[:, 5:6].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            #break  # time limit exceeded

    return output


def pred_out(pred, imgsz):
    feats, pred_distri, confidence, pkpts, pred_kpt = pred
    
    #process bbox
    confidence = confidence.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    stride = [imgsz/x.size(2) for x in feats]
    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
    pred_bboxes = bbox_decode_xywh(anchor_points, pred_distri) * stride_tensor #ok
    confidence = confidence.sigmoid()
    
    #y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)
    #input concat          #0123          #4         #5        #678
    nms_input = torch.cat([pred_bboxes, confidence, confidence, pred_kpt],dim = -1) #xywh in 640 scale
    #input in yolopose xywh-kpt in 640 scale
    preds = non_max_suppression(nms_input,
                                conf_thres = 0.05,
                                iou_thres=0.45,
                                agnostic=False) #nms output is xyxy
    return preds, torch.cat([pred_bboxes, confidence, pred_kpt],dim = -1)

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            
    return torch.device('cuda:0' if cuda else 'cpu')


def test(model, dataloader, compute_loss=False, half_precision=False, plot = False, txtname = None, epoch = None, doeval = False):
    device = next(model.parameters()).device

    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()
    model.set_inference()
    
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader))
    sa,sb,sx,sy = 'radar_images', 'images', '.png','.jpg'

    gts = []
    pds = []

    for i, (img, targets, paths, shapes) in pbar:
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            pred = model(img)
            if compute_loss:
                loss, loss_items = compute_loss(pred, targets)
            outs, pds_item = pred_out(pred,height)  #torch.Size([1, 57])

            if plot:

                for out,imname,shape in zip(outs, paths, shapes):
                    imname = imname.replace(sa,sb).replace(sx,sy)
                    im = cv2.imread(imname)
                    for o in out:
                        o = o.view(1,-1)
                        scale_coords(img.shape[2:], o[:,:4], shape[0], shape[1], kpt_label=False)
                        scale_coords(img.shape[2:], o[:,6:], shape[0], shape[1], kpt_label=True, step=3)
                    
                        plot_skeleton_kpts(im,o[...,6:].permute(1,0))
                    #cv2.imshow("a", im)       
                    #cv2.waitKey(150)
                    cv2.imwrite('/yq/RE/CAttHRnetPose/pred_test06/'+imname.split('/')[-1], im)

        gts.append(targets[:,2:])
        pds.append(torch.cat((outs[0][:,:5],outs[0][:,6:]),dim=1))

    if doeval:
        if epoch is not None:
            with open(txtname+'test.txt','a') as f:
                f.write('EPOCH: '+ str(epoch) +  '\n')

        eval = Eval(gts,pds,'keypoints',txtname)
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
            
                
def inference(model, dataloader, half_precision=False, plot = False):
    device = next(model.parameters()).device

    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()
    model.set_inference()
    
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader))
    sa,sb,sx,sy = 'radar_images', 'images', '.png','.jpg'


    for i, (img, paths, shapes) in pbar:
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            pred = model(img)

            outs, pds_item = pred_out(pred,height)  #torch.Size([1, 57])

            if plot:

                for out,imname,shape in zip(outs, paths, shapes):
                    imname = imname.replace(sa,sb).replace(sx,sy)
                    im = cv2.imread(imname)
                    for o in out:
                        o = o.view(1,-1)
                        scale_coords(img.shape[2:], o[:,:4], shape[0], shape[1], kpt_label=False)
                        scale_coords(img.shape[2:], o[:,6:], shape[0], shape[1], kpt_label=True, step=3)
                    
                        plot_skeleton_kpts(im,o[...,6:].permute(1,0))
                    #cv2.imshow("a", im)       
                    #cv2.waitKey(150)
                    cv2.imwrite('/yq/RE/CAttHRnetPose/pred_test06/'+imname.split('/')[-1], im)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str, default='cfg/model_small3.yaml', help='model.yaml path')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--data', type=str, default='cfg/data_06.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--pretrained', type=str, default='trains/train0613_1/last.pt', help='pretrained model')   #train0523_1
    opt = parser.parse_args()
    with open(opt.cfg) as f:
        opt.cfg = yaml.safe_load(f)
    with open(opt.hyp) as f:
        opt.hyp = yaml.safe_load(f)
    device = select_device(opt.device, batch_size=None)
    model = AttHRnetPose(opt.cfg['MODEL']).to(device)
    if os.path.isfile(opt.pretrained):
        model=torch.load(opt.pretrained)

    is_inference=False

    if is_inference:

        #data
        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)  # data dict
        data_path = data_dict['val']
        cache_path = data_dict['cache_path']
        
        dataset = LoadImages(data_path, cache_path, img_size=640, hyp=opt.hyp)

        '''dataL = int(len(dataset))
        generator = torch.Generator().manual_seed(SEED)
        trainset, testset = random_split(
            dataset=dataset,
            lengths=[dataL - 500, 500],
            generator=generator
        )'''
        testset = dataset
        #sampler = torch.utils.data.distributed.DistributedSampler(trainset) if opt.local_rank != -1 else None
        loader = torch.utils.data.DataLoader
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=1,
                                                num_workers=0,
                                                sampler=None,
                                                pin_memory=True,
                                                collate_fn=LoadImages.collate_fn,
                                                shuffle = False)
        
        #test(model, testloader, compute_loss=False, half_precision=False, plot =True,doeval=False)
        inference(model,testloader, half_precision=False, plot = True)
    else:

        
    #data
        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)  # data dict
        data_path = data_dict['path']
        cache_path = data_dict['cache_path']
        

        dataset = LoadImagesAndLabels(data_path, cache_path, img_size=640, hyp=opt.hyp)

        dataL = int(len(dataset))
        generator = torch.Generator().manual_seed(SEED)
        trainset, testset = random_split(
            dataset=dataset,
            lengths=[dataL - 4000, 4000],
            generator=generator
        )

        
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=1,
                                                num_workers=4,
                                                sampler=None,
                                                pin_memory=True,
                                                collate_fn=LoadImagesAndLabels.collate_fn,
                                                shuffle = False)    

        
        test(model, testloader, compute_loss=False, half_precision=False, plot =True,doeval=True)

    









    
