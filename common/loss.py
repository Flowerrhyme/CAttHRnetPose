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
        self.n_kpt = model.detect.nkpt
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
        self.no = model.detect.no
        self.no_kpt = model.detect.no_kpt

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0
        
        loss = torch.zeros(5, device=device)  # box, dfl, kpt, kptv, confidence
        feats, _, _, _ = p #x, box, confidence,kpt
        batch_size = feats[0].shape[0]
        #new
        for feat in feats:
            # g = w and h in feat
            pred_distri, pred_scores, pkpts = feat.view(batch_size, self.no, -1).split((self.reg_max * 4, 1,self.no_kpt), 1)
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            stride = [self.imgsz[0]/feat.size(2)]


            anchor_points, stride_tensor = make_anchors([feat], stride, 0.5)
            gt = self.preprocess(targets, batch_size, scale_tensor=torch.tensor(self.imgsz)[[1, 0, 1, 0]].to(device))
            gt_labels, gt_bboxes, gt_kpts = gt.split((1, 4, 34), 2)  # cls,   xyxy gt_box:640,   gt_kpt:0-1
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri) # g scale xyxy

            # Task align 
                           #0-640 xyxy                             0-1
            target_labels, target_bboxes, target_scores, fg_mask, target_kpts = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  #0-640
                anchor_points * stride_tensor, #0-640
                gt_labels,
                gt_bboxes, #0-640 xyxy
                mask_gt,
                gt_kpts)
            
            # grid align 
            target_bboxes, target_kpts, target_scores,fg_mask =self.target_gij(
                feat,
                gt_bboxes,
                stride[0],
                fg_mask,
                mask_gt,
                gt_kpts,
                target_bboxes,
                target_kpts,
                target_scores)
            target_x = target_kpts[...,0::2]
            zero_mask = (target_x>0)
            
            #target kpt -= grid center
            target_kpts = self.build_tkpt(anchor_points, target_kpts, feat.size(2))

            target_bboxes /= stride_tensor #in g scale 
            target_scores_sum = target_scores.sum()

            #compute loss
            if fg_mask.sum():   #box loss , dfl loss
                box_loss, dfl_loss, iou = self.bbox_loss(pred_distri,
                                                    pred_bboxes,  #in g scale 
                                                    anchor_points,
                                                    target_bboxes, #in g scale
                                                    target_scores,
                                                    target_scores_sum,
                                                    fg_mask)  #box loss , dfl loss
                loss[0] += box_loss
                loss[1] += dfl_loss
                #kpt loss
                pkpts = pkpts.permute(0,2,1)
                pkpt_x = pkpts[:, :, 0::3] * 2. - 0.5
                pkpt_y = pkpts[:, :, 1::3] * 2. - 0.5
                pkpt_score = pkpts[:,:, 2::3]
                kpt_mask = fg_mask.unsqueeze(-1).repeat((1,1,17))
                kpt_mask = kpt_mask & zero_mask
                loss[3] += self.BCEkptv(pkpt_score, kpt_mask.float()) 
                target_bboxes_w,target_bboxes_h = target_bboxes[:,:, 2] - target_bboxes[:,:, 0] ,target_bboxes[:,:,3] - target_bboxes[:,:,1]
                d = (pkpt_x-target_kpts[:,:,0::2])**2 + (pkpt_y-target_kpts[:,:,1::2])**2
                s = (target_bboxes_w*target_bboxes_h).unsqueeze(-1)
                kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                loss[2] += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()

                #object loss
                
                loss[4] += self.BCEobj(pred_scores, target_scores).sum() / target_scores_sum

        #new end
        out_loss = sum(loss)

        return out_loss * batch_size , torch.cat((loss,out_loss.unsqueeze(0)))
    
    def target_gij(self, feat,gt_bboxes,stride,fg_mask,mask_gt,gt_kpts,target_bboxes,target_kpts,target_scores):
        device = gt_bboxes.device
        g = feat.size(2)
        gt_bboxes_xyxy = gt_bboxes #640 scale
        gt_bboxes = xyxy2xywh(gt_bboxes)
        gt_bboxes /= stride #in g scale
        off = torch.tensor([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                                ], device=gt_bboxes.device).float() 
        gij = gt_bboxes[:,:,:2] #bs , num_max, xywh
        gij = gij.unsqueeze(2).repeat((1,1,5,1))
        off = off.unsqueeze(0).unsqueeze(0)
        gij -= off
        gi,gj = gij[...,0].long(),gij[...,1].long()
        gi = gi.clamp_(0, g - 1)
        gj = gj.clamp_(0, g - 1)
        fg_mask = fg_mask.view(-1,g,g)
        for i in range(gi.size(1)):
            gij_mask = torch.zeros_like(fg_mask).bool()
            gij_mask = gij_mask.view(-1,g,g)
            #batch index, gridx index, gridy index
            maskgt = mask_gt[:,i,:].repeat(1,5).bool()
            batch_idx = torch.range(0,gij_mask.shape[0]-1).long().view(gij_mask.shape[0],1).repeat(1,5).to(device)
            gridx = gi[:,i,:]
            gridy = gj[:,i,:]
            batch_idx = torch.masked_select(batch_idx, maskgt).flatten()
            gridx = torch.masked_select(gridx, maskgt).flatten()
            gridy = torch.masked_select(gridy, maskgt).flatten()
            #grid mask
            gij_mask[batch_idx,gridx,gridy]=True

            #align
            mask = ((fg_mask == False) & (gij_mask==True)).view(-1,g*g,1)
            fg_mask = fg_mask | gij_mask
            
            gt_bx = gt_bboxes_xyxy[:,i,:]
            gxy_target_box = gt_bx.view(-1,1,1,4).repeat(1,g,g,1).view(-1,g*g,4)
            target_bboxes = torch.where(mask.repeat(1,1,4), gxy_target_box, target_bboxes)

            gt_kp = gt_kpts[:,i,:]
            gxy_target_kpts = gt_kp.view(-1,1,1,self.n_kpt*2).repeat(1,g,g,1).view(-1,g*g,self.n_kpt*2)
            target_kpts = torch.where(mask.repeat(1,1,self.n_kpt*2), gxy_target_kpts, target_kpts)

            target_scores = torch.where(mask, 0.8, target_scores)
                #g scale xyxy
        return target_bboxes,target_kpts, target_scores,fg_mask.view(-1,g*g)
      

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


    def build_tkpt(self, anchor_points, target_kpts, gain): #targets:(batchsize,12800,34)

        anchor_points = anchor_points.view(1,-1,2).repeat(target_kpts.size(0),1,17)
        target_kpts *= gain  # g scale
        target_kpts -= anchor_points
        return target_kpts
        


       

    



            




        

                


