# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        print('batch',score.size(0))
        print('ph',ph)
        print('pw',pw)
        h, w = target.size(1), target.size(2) #ground truth
        print('The same time h is ',h)
        print(w)
        print('batch',target.size(0))
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        # print(pred)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0
        # print('tmp_target',tmp_target)
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        # print('Also')
        # print(pred)
        # print(ind)
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()



class SeedLoss(nn.Module):
    def __init__(self, ignore_label=-1,n_sigma=1,weight=None):
        super().__init__()

        self.ignore_label = ignore_label
        self.n_sigma=n_sigma
        #EDO UELEI ALLAGI ME BASI TI DIASTASI POU EXEI KAUE FORA
        # coordinate map
        # xm = torch.linspace(0, 2, 2048).view(
        #     1, 1, -1).expand(1, 1024, 2048)
        # ym = torch.linspace(0, 1, 1024).view(
        #     1, -1, 1).expand(1, 1024, 2048)
        # xym = torch.cat((xm, ym), 0)
        # self.register_buffer("xym", xym)

        #cross entropy
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')


    def forward(self,score,target,w_f=1, w_pixel=1, **kwargs):
        batch_size,ph, pw = score.size(0), score.size(2), score.size(3) #batch size to check
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')

        xm = torch.linspace(0, 2, h).view(
            1, 1, -1).expand(1, w, h)
        ym = torch.linspace(0, 1, w).view(
            1, -1, 1).expand(1, w, h)
        xym = torch.cat((xm, ym), 0)
        xym_s = xym[:, 0:h, 0:w].contiguous() # 2 x h x w
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0 #ground truth
        f_loss=0
        loss=0
        #pred = F.softmax(score, dim=1) # maybe useless
        for b in range(0,batch_size):
            # sos
            spatial_emb = torch.tanh(score[b, 0:2]) + xym_s  # 2 x h x w  -> seed pixel
            f_map = torch.sigmoid(
                score[b, 2 + self.n_sigma:2 + self.n_sigma + 1]) # 1 x h x w
            # confidence map loss
            if tmp_target[b,0:2] != tmp_target[spatial_emb]:
                f_loss-=torch.log(1-f_map)
            else:
                f_loss+=torch.log(f_map)
            # cross entropy losses for seed and confidence
            pixel_losses = self.criterion(score, target).contiguous().view(-1)
            loss+=w_f*f_loss + w_pixel*pixel_losses




