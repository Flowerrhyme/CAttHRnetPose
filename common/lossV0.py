import torch
import torch.nn as nn
import math
from common.general import bbox_iou, xywh2xyxy, xyxy2xywh
from common.assigner import TaskAlignedAssigner
from common.general import dist2bbox, make_anchors, bbox2dist
import torch.nn.functional as F


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        #loss_iou *= bbox_weight
        #loss_iou = loss_iou.sum() / target_scores_sum
        loss_iou = loss_iou.mean()

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

class ComputeLoss:
    def __init__(self, opt, model, use_dfl=True):
        self.gr = 0.5
        device = next(model.parameters()).device
        self.device = device
        self.nc = 1
        self.nl = opt.cfg['MODEL']['OUT_NUM']
        self.imgsz = 2*[opt.img_size[0]]
        self.BCEkptv = nn.BCEWithLogitsLoss()
        self.BCEobj = nn.BCEWithLogitsLoss()
        self.assigner = TaskAlignedAssigner(topk= 10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6)
        self.reg_max = 16
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(16).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0
        
        loss = torch.zeros(5, device=device)  # box, dfl, kpt, kptv, confidence
        feats, pred_distri, confidence, pkpts = p #x, box, confidence, cls,kpt
        batch_size, _ = confidence.shape[:2]
        pred_scores = torch.ones_like(confidence.permute(0, 2, 1).contiguous()).to(device)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        stride = [self.imgsz[0]/x.size(2) for x in feats]

        #bbox loss
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
        targets = self.preprocess(targets, batch_size, scale_tensor=torch.tensor(self.imgsz)[[1, 0, 1, 0]].to(device))
        gt_labels, gt_bboxes, gt_kpts = targets.split((1, 4, 34), 2)  # cls, xyxy gt_box:640 gt_kpt:0-1
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri) #320 160 xyxy
        
                        #0-640 xyxy
        target_labels, target_bboxes, target_scores, fg_mask, target_kpts = self.assigner(
            pred_scores,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  #0-640
            anchor_points * stride_tensor, #0-640
            gt_labels,
            gt_bboxes, #0-640 xyxy
            mask_gt,
            gt_kpts)
        
        #yolov7 anchor align
        #self.target_gij(feats,gt_bboxes,stride)

        target_bboxes /= stride_tensor #in different scale 320,120
        target_scores_sum = target_scores.sum()

        if fg_mask.sum():   #box loss , dfl loss
            loss[0], loss[1], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,  #in different scale 320,120
                                                   anchor_points,
                                                   target_bboxes, #in different scale 320,120
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)  #box loss , dfl loss
            #loss[0] *= 7.5
            #kpt loss
            pkpts = pkpts.permute(0,2,1)
            pkpt_x = pkpts[:, :, 0::3] * 2. - 0.5
            pkpt_y = pkpts[:, :, 1::3] * 2. - 0.5
            pkpt_score = pkpts[:,:, 2::3]
            #kpt_mask = (target_kpts[:,:, 0::2] != 0) & fg_mask.unsqueeze(-1).repeat((1,1,17))
            kpt_mask = fg_mask.unsqueeze(-1).repeat((1,1,17))
            loss[3] += self.BCEkptv(pkpt_score, kpt_mask.float()) 
            target_bboxes_w,target_bboxes_h = target_bboxes[:,:, 2] - target_bboxes[:,:, 0] ,target_bboxes[:,:,3] - target_bboxes[:,:,1]

            #build tkpt
            target_kpts = self.build_tkpt(feats, target_bboxes, target_kpts)

            # pkpt_x size(batchsize,128000,17)
            d = (pkpt_x-target_kpts[:,:,0::2])**2 + (pkpt_y-target_kpts[:,:,1::2])**2
            s = (target_bboxes_w*target_bboxes_h).unsqueeze(-1)
            kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
            loss[2] = kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()
        
        self.target_gij(feats,gt_bboxes,stride,fg_mask,mask_gt)


        #object loss
        index = torch.where(fg_mask)
        all_iou = torch.zeros_like(confidence).squeeze(1)
        all_iou[index] = (1.0 - self.gr) + self.gr * (iou.squeeze(1)).clamp(0).type(loss.dtype)
        loss[4] = self.BCEobj(confidence.squeeze(1), all_iou)



        out_loss = sum(loss)

        return out_loss * batch_size , torch.cat((loss,out_loss.unsqueeze(0)))

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def preprocess(self,targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 39, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 39, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))

        return out  #size[batchsize, counts.max, no]


    def build_tkpt(self, feats, target_bboxes, target_kpts:torch.Tensor): #targets:(batchsize,12800,34)
        device = target_bboxes.device
        whs = tuple([x.size(2)**2 for x in feats])
        wh = tuple([x.size(2) for x in feats]) 
        kpt_ps = target_kpts.split(whs, dim = 1)  #kpt per image_scale
        target_bboxes = xyxy2xywh(target_bboxes)
        bbox_ps = target_bboxes.split(whs, dim = 1)
        out = []
        for g, kpt,bbox in zip(wh, kpt_ps,bbox_ps):
            nkpt = kpt.size(2)//2
            
            kpt_gain = torch.tensor([g])[nkpt*[0, 0]].to(device)  # xyxy gain
            gxy = bbox[:, :, 0:2]
        
            gij = gxy.long()
            kpt *= kpt_gain
            kpt -= gij.repeat(1,1,nkpt)
            out.append(kpt)
        return torch.cat(out, dim = 1)

    def target_gij(self, feats,gt_bboxes,strides,fg_mask,mask_gt):
        wh = tuple([x.size(2) for x in feats]) 
        whs = tuple([x.size(2)**2 for x in feats])
        gt_bboxes = xyxy2xywh(gt_bboxes)
        fg_mask = fg_mask.split(whs, dim = 1)
        gis = []
        gjs = []
        st = 0
        num_max = mask_gt.size(1)
        for g,stride,fgmask in zip(wh,strides,fg_mask):
            fgmask =fgmask.view(-1,g,g)
            gt_bboxes /= stride
            gij = gt_bboxes[:,:,:2]
            #g = 1 # bias
            off = torch.tensor([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                                ], device=gt_bboxes.device).float() 
            ofL = off.size(0)
            gij = gij.unsqueeze(2).repeat((1,1,5,1))
            off = off.unsqueeze(0).unsqueeze(0)
            gij -= off
            gi,gj = gij[...,0].long(),gij[...,1].long()
            gi = gi.clamp_(0, g - 1)
            gj = gj.clamp_(0, g - 1)
          



            




        

                


