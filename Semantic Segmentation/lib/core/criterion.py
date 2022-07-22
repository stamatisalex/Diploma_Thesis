# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.functions_plane import *

# seed = 3
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor
# device = torch.device('cuda:{}'.format(args.local_rank))
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        #score-> prediction
        #target->label
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
    
    def forward(self, s_i, target,flag=True, **kwargs):
        ph, pw = s_i.size(2), s_i.size(3)
        h, w = target.size(1), target.size(2) #ground truth
        if ph != h or pw != w:
            s_i = F.upsample(input=s_i, size=(h, w), mode='bilinear')
        if(flag):
            # if (debug):
            #     print("-1", True in torch.isinf(s_i))
            pred = F.softmax(s_i, dim=1)
            # if (debug):
            #     print("0", True in torch.isinf(pred))
        else:
            pred = s_i
        pixel_losses = self.criterion(s_i, target).contiguous().view(-1) # batch * 512 * 1024 ~ 100000
        # if(debug):
        #     print("1",True in torch.isinf(pixel_losses))
        mask = target.contiguous().view(-1) != self.ignore_label   #The above line creates a Binary tensor that has a False at each place for the value=ignore_index
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        # if (debug):
        #     print("2",True in torch.isinf(pixel_losses))
        pixel_losses = pixel_losses[pred < threshold]
        # if (debug):
        #     print("3",True in torch.isinf(pixel_losses))
        return pixel_losses.mean()
        #
        # if (torch.isnan(result)):
        #     torch.save(pixel_losses, 'tensor.pt')
        #     print("NaN detected in {}".format(name))
        #     return torch.mean(pixel_losses[~torch.isnan(pixel_losses)])
        # else:
        #     return result





class Confidence_Loss(nn.Module):
    def __init__(self, device, ignore_label=-1):
        super().__init__()
        self.ignore_label = ignore_label
        self.device = device
        ##########################################################
        # coordinate map
        # w=1024
        # h=512
        # x -> [0, ... ,w] , y-> [0, ... , h]


        #Uncomment this

        #
        # xm = torch.linspace(0, 1023, 1024).view(
        #     1, 1, -1).expand(1, 512, 1024)     # 1 x 512 x 1024
        # ym = torch.linspace(0, 511, 512).view(
        #     1, -1, 1).expand(1, 512, 1024)
        # xym = torch.cat((xm, ym), 0)          # 1 x 512 x 1024
        # self.register_buffer("xym", xym)
        ############################################################
        # xm = torch.linspace(0, 2, 1024).view(
        #     1, 1, -1).expand(1, 512, 1024)
        # ym = torch.linspace(0, 1, 512).view(
        #     1, -1, 1).expand(1, 512, 1024)
        # xym = torch.cat((xm, ym), 0)
        # self.register_buffer("xym", xym)


    def calculate_H_seed(self,target,x,y):

        # floor function in order to do nearest neighbor algorithm
        x = torch.floor(x).type(dtype_long)
        y = torch.floor(y).type(dtype_long)

        x = torch.clamp(x, 0, target.shape[2] - 1)
        y = torch.clamp(y, 0, target.shape[1] - 1)

        return target[:,y,x]


    def forward(self,o_f,target, w_f=1, **kwargs):
        batch_size,ph, pw = o_f.size(0), o_f.size(2), o_f.size(3) #batch size to check
        h, w = target.size(1), target.size(2)  # h->512 , w->1024

        # coordinate map
        # w=1024
        # h=512
        # x -> [0, ... ,w] , y-> [0, ... , h]

        xm = torch.linspace(0, w-1, w).view(
            1, 1, -1).expand(1, h, w)     # 1 x h x w
        ym = torch.linspace(0, h-1, h).view(
            1, -1, 1).expand(1, h, w)
        xym = torch.cat((xm, ym), 0)          # 1 x h x w
        xym = xym.to(self.device)
        # print('before',o_f.size())
        if ph != h or pw != w:
            o_f = F.upsample(input=o_f, size=(h, w), mode='bilinear')

        xym_s = xym[:, 0:h, 0:w].contiguous()  # 2 x h x w
        tmp_target = target.clone() # batch x h x w
        tmp_target[tmp_target == self.ignore_label] = 0 #ground truth
        ####################################

        # first way


        # f_loss=0
        # f=o_f[:,2]
        # spatial_pix = o_f[:, 0:2] + xym_s
        #
        # x_cords = spatial_pix[:,0] # batch x h x w
        # y_cords = spatial_pix[:,1] # batch x h x w
        #
        # H_s = torch.ones(tmp_target.size()) # batch x h x w
        # for b in range(0,batch_size):
        #     H_s[b] = self.calculate_H_seed(tmp_target[b].unsqueeze(0),x_cords[b],y_cords[b])
        # H_s = H_s.type(dtype_long)
        # mask = tmp_target == H_s # batch x h x w
        # mask2 = mask < 1  # logical not
        # f_loss = (torch.sum(-torch.log(f[mask])) + torch.sum(-torch.log(1 - f[mask2])))/(h*w)
        # loss += w_f * f_loss.mean()

        ######################################

        f_loss = 0
        eps = 1e-7
        # print('after',o_f.size())
        for b in range(0,batch_size):
            f = o_f[b, 2]  # h x w
            spatial_pix = o_f[b, 0:2] + xym_s  # 2 x h x w
            # print('f',f)

            #Scaling

            x_cords = spatial_pix[0]  # h x w
            y_cords = spatial_pix[1]  # h x w

            #Target map ofsset vectors prediction
            H_s = self.calculate_H_seed(tmp_target[b].unsqueeze(0),x_cords,y_cords) # 1 x h x w

            mask = tmp_target[b] == H_s.squeeze(0) # h x w
            mask2 = mask < 1 # logical not
            f_loss+= (torch.sum(-torch.log(f[mask]+eps)) + torch.sum(-torch.log(1-f[mask2]+eps))) / (h*w)

        f_loss=f_loss/(b+1)
        loss = w_f * f_loss

        return loss

class Confidence_Loss_2(nn.Module):
    def __init__(self,ignore_label=-1):
        super().__init__()
        self.ignore_label = ignore_label
        self.get_coords = get_coords

    def forward(self,offset,f,target, **kwargs):
        batch_size,ph, pw = f.size(0), f.size(2),f.size(3)  # batch size to check
        h, w = target.size(1), target.size(2)  # h->512 , w->1024


        # print('before',o_f.size())
        if ph != h or pw != w:
            f = F.upsample(input=f, size=(h, w), mode='bilinear')
            offset = F.upsample(input=offset, size=(h, w), mode='bilinear')

        coords = self.get_coords(batch_size, h, w, fix_axis=True)
        ocoords_orig = nn.Parameter(coords, requires_grad=False)

        mask_initial = target != self.ignore_label # batch x h x w
        tmp_target = target.clone()  # batch x h x w
        tmp_target[tmp_target == self.ignore_label] = 0  # ground truth
        tmp_target = tmp_target.type(dtype)
        eps = 1e-7
        # f --> batch x 1 x h x w
        # offset --> batch x 2 x h x w
        offset = offset.permute(0, 2, 3 ,1) # batch x h x w x 2
        ocoords = ocoords_orig + offset # batch x h x w x 2

        # ocoords = o_f[:, 0:2]  # batch x 2 x h x w
        # ocoords = ocoords.permute(0, 2, 3 ,1) # batch x h x w x 2

        H_s = F.grid_sample(tmp_target.unsqueeze(1), ocoords,mode='nearest', padding_mode='border') # batch x 1 x h x w
        mask = tmp_target.unsqueeze(1) == H_s
        mask2 = mask < 1  # logical not
        # f_loss = (torch.sum(-torch.log(f[mask & mask_initial.unsqueeze(1)] + eps)) + torch.sum(-torch.log(1 - f[mask2 & mask_initial.unsqueeze(1)] + eps))) / (h * w)
        f_loss = (torch.sum(-torch.log(f[mask] + eps)) + torch.sum(-torch.log(1 - f[mask2 ] + eps))) / (h * w)

        return f_loss.mean()


class Size_Loss(nn.Module):
    def __init__(self,treshold=0.1):
        super().__init__()
        self.treshold = torch.tensor(treshold)

    def forward(self,offset, **kwargs):
        batch,_,h,w=offset.size()
        offset = offset.reshape(batch,2,h*w)
        offset = torch.norm(offset,dim=1)
        maximum =torch.maximum((torch.norm(offset,dim=1) - self.treshold),torch.tensor(0))
        return torch.pow(maximum,2)
























#
#
# class SeedLoss(nn.Module):
#     def __init__(self, device, ignore_label=-1, thres=0.7,
#         min_kept=100000, weight=None):
#         super().__init__()
#         self.ignore_label = ignore_label
#         self.thresh=thres
#         self.min_kept=max(1, min_kept)
#         self.device = device
#         ##########################################################
#         # coordinate map
#         # w=1024
#         # h=512
#         # x -> [0, ... ,w] , y-> [0, ... , h]
#
#
#         #Uncomment this
#
#         #
#         # xm = torch.linspace(0, 1023, 1024).view(
#         #     1, 1, -1).expand(1, 512, 1024)     # 1 x 512 x 1024
#         # ym = torch.linspace(0, 511, 512).view(
#         #     1, -1, 1).expand(1, 512, 1024)
#         # xym = torch.cat((xm, ym), 0)          # 1 x 512 x 1024
#         # self.register_buffer("xym", xym)
#         ############################################################
#         # xm = torch.linspace(0, 2, 1024).view(
#         #     1, 1, -1).expand(1, 512, 1024)
#         # ym = torch.linspace(0, 1, 512).view(
#         #     1, -1, 1).expand(1, 512, 1024)
#         # xym = torch.cat((xm, ym), 0)
#         # self.register_buffer("xym", xym)
#
#         #Ohem cross entropy
#         # self.criterion = OhemCrossEntropy(ignore_label=ignore_label,
#         #                              thres=thres,
#         #                              min_kept=min_kept,
#         #                              weight=weight)
#         self.criterion = nn.CrossEntropyLoss(weight=weight,
#                                              ignore_index=ignore_label,
#                                              reduction='none')
#
#     def calculate_H_seed(self,target,x,y):
#
#         # floor function in order to do nearest neighbor algorithm
#         x = torch.floor(x).type(dtype_long)
#         y = torch.floor(y).type(dtype_long)
#
#         x = torch.clamp(x, 0, target.shape[2] - 1)
#         y = torch.clamp(y, 0, target.shape[1] - 1)
#
#         return target[:,y,x]
#
#
#     def forward(self,o_f,s_s,s_f,target, w_s_s=1, w_s_f=1, w_f=1, **kwargs):
#         batch_size,ph, pw = o_f.size(0), o_f.size(2), o_f.size(3) #batch size to check
#         h, w = target.size(1), target.size(2)  # h->512 , w->1024
#
#         # coordinate map
#         # w=1024
#         # h=512
#         # x -> [0, ... ,w] , y-> [0, ... , h]
#
#         xm = torch.linspace(0, w-1, w).view(
#             1, 1, -1).expand(1, h, w)     # 1 x h x w
#         ym = torch.linspace(0, h-1, h).view(
#             1, -1, 1).expand(1, h, w)
#         xym = torch.cat((xm, ym), 0)          # 1 x h x w
#         xym = xym.to(self.device)
#         # print('before',o_f.size())
#         if ph != h or pw != w:
#             s_s = F.upsample(input=s_s, size=(h, w), mode='bilinear')
#             s_f = F.upsample(input=s_f, size=(h, w), mode='bilinear')
#             o_f = F.upsample(input=o_f, size=(h, w), mode='bilinear')
#
#         #Ohem cross entropy
#
#         loss_s_s = self.criterion(s_s, target).contiguous().view(-1)
#         loss_s_f = self.criterion(s_f, target).contiguous().view(-1)
#
#         # for cross entropy comment it
#         mask = target.contiguous().view(-1) != self.ignore_label
#
#
#         # loss_s_s = self.criterion(s_s, target)
#         # loss_s_f = self.criterion(s_f, target)
#
#         xym_s = xym[:, 0:h, 0:w].contiguous()  # 2 x h x w
#         tmp_target = target.clone() # batch x h x w
#         tmp_target[tmp_target == self.ignore_label] = 0 #ground truth
#
#         # if you use only cross_entropy place in comments the following
#
#         pred_s_s = s_s.gather(1, tmp_target.unsqueeze(1))
#         pred_s_f = s_f.gather(1, tmp_target.unsqueeze(1))
#         pred_s_s, ind_s_s = pred_s_s.contiguous().view(-1, )[mask].contiguous().sort()
#         pred_s_f, ind_s_f = pred_s_f.contiguous().view(-1, )[mask].contiguous().sort()
#
#         min_value_s_s = pred_s_s[min(self.min_kept, pred_s_s.numel() - 1)]
#         min_value_s_f = pred_s_f[min(self.min_kept, pred_s_f.numel() - 1)]
#         threshold_s_s = max(min_value_s_s, self.thresh)
#         threshold_s_f = max(min_value_s_f, self.thresh)
#
#         loss_s_s = loss_s_s[mask][ind_s_s]
#         loss_s_f = loss_s_f[mask][ind_s_f]
#         loss_s_s = loss_s_s[pred_s_s < threshold_s_s]
#         loss_s_f = loss_s_f[pred_s_f < threshold_s_f]
#
#         # till here
#
#         #losses
#         loss= w_s_s * loss_s_s.mean() + w_s_f * loss_s_f.mean()
#         # loss = w_s_s * loss_s_s + w_s_f * loss_s_f
#
#         ####################################
#
#         # first way
#
#
#         # f_loss=0
#         # f=o_f[:,2]
#         # spatial_pix = o_f[:, 0:2] + xym_s
#         #
#         # x_cords = spatial_pix[:,0] # batch x h x w
#         # y_cords = spatial_pix[:,1] # batch x h x w
#         #
#         # H_s = torch.ones(tmp_target.size()) # batch x h x w
#         # for b in range(0,batch_size):
#         #     H_s[b] = self.calculate_H_seed(tmp_target[b].unsqueeze(0),x_cords[b],y_cords[b])
#         # H_s = H_s.type(dtype_long)
#         # mask = tmp_target == H_s # batch x h x w
#         # mask2 = mask < 1  # logical not
#         # f_loss = (torch.sum(-torch.log(f[mask])) + torch.sum(-torch.log(1 - f[mask2])))/(h*w)
#         # loss += w_f * f_loss.mean()
#
#         ######################################
#         f_loss=0
#         eps = 1e-7
#         print('after',o_f.size())
#         for b in range(0,batch_size):
#             f = o_f[b, 2]  # h x w
#             spatial_pix = o_f[b, 0:2] + xym_s  # 2 x h x w
#             # print('f',f)
#
#
#             #Scaling
#
#             x_cords = spatial_pix[0]  # h x w
#             y_cords = spatial_pix[1]  # h x w
#
#             #Target map ofsset vectors prediction
#             H_s = self.calculate_H_seed(tmp_target[b].unsqueeze(0),x_cords,y_cords) # 1 x h x w
#
#             mask = tmp_target[b] == H_s.squeeze(0) # h x w
#             mask2 = mask < 1 # logical not
#             f_loss+= (torch.sum(-torch.log(f[mask]+eps)) + torch.sum(-torch.log(1-f[mask2]+eps))) / (h*w)
#
#         f_loss=f_loss/(b+1)
#         loss += w_f * f_loss
#
#         return loss



















    #Slow solution

    # def forward(self,o_f,s_s,s_f,target, w_s_s=1, w_s_f=1, w_f=1, **kwargs):
    #     batch_size,ph, pw = s_f.size(0), s_f.size(2), s_f.size(3) #batch size to check
    #     h, w = target.size(1), target.size(2)  # h->512 , w->1024
    #     if ph != h or pw != w:
    #         s_s = F.upsample(input=s_s, size=(h, w), mode='bilinear')
    #         s_f = F.upsample(input=s_f, size=(h, w), mode='bilinear')
    #         o_f = F.upsample(input=o_f, size=(h, w), mode='bilinear')
    #     # print('s_s',s_s.size())
    #     loss_s_s = self.criterion(s_s, target).contiguous().view(-1)
    #     loss_s_f = self.criterion(s_f, target).contiguous().view(-1)
    #
    #     xym_s = self.xym[:, 0:h, 0:w].contiguous()  # 2 x h x w
    #     tmp_target = target.clone() # batch x h x w
    #     tmp_target[tmp_target == self.ignore_label] = 0 #ground truth
    #
    #     #losses
    #
    #     loss= w_s_s * loss_s_s.mean() + w_s_f * loss_s_f.mean()
    #
    #     for b in range(0,batch_size):
    #         # print('s_s_b', s_s[b].size())
    #         # print('t',target[b].size())
    #         spatial_pix = o_f[b, 0:2] + xym_s  # 2 x h x w
    #         f = torch.sigmoid(o_f[b, 2])
    #
    #         #scaling
    #         x_cords = w * spatial_pix[0]  # h x w
    #         y_cords = h * spatial_pix[1]  # h x w
    #
    #         tmp_target_seed = torch.ones(tmp_target[b].size())
    #         tmp_target_seed = tmp_target_seed.type(dtype_long)
    #         f_loss = 0
    #         for y in range(h):
    #             for x in range(w):
    #
    #                 #nearest neighboor
    #                 i = torch.floor(x_cords[y][x])
    #                 j = torch.floor(y_cords[y][x])
    #                 #convert to long
    #                 i=i.long()
    #                 j=j.long()
    #
    #                 # Grid Limits
    #                 if (i < 0):
    #                     i = 0
    #                 elif (i > (w-1)):
    #                     i = w - 1
    #                 if (j < 0):
    #                     j = 0
    #                 elif (j > (h-1)):
    #                     j = h - 1
    #
    #                 tmp_target_seed[y][x] = tmp_target[b][j][i]
    #                 if tmp_target[b][y][x] != tmp_target_seed[y][x]:
    #                     f_loss-=torch.log(1-f[y][x])
    #                 else:
    #                     f_loss-=torch.log(f[y][x])
    #         # cross entropy losses
    #         # loss_s_i_2 = self.criterion(o, target).contiguous().view(-1)
    #
    #         loss += w_f * f_loss
    #     print('seed loss',loss)
    #     return loss

















































# class SeedLoss(nn.Module):
#     def __init__(self, ignore_label=-1,n_sigma=1,weight=None):
#         super().__init__()
#         self.ignore_label = ignore_label
#         self.n_sigma=n_sigma
#         #EDO UELEI ALLAGI ME BASI TI DIASTASI POU EXEI KAUE FORA
#         # coordinate map
#         xm = torch.linspace(0, 2, 1024).view(
#             1, 1, -1).expand(1, 512, 1024)
#         ym = torch.linspace(0, 1, 512).view(
#             1, -1, 1).expand(1, 512, 1024)
#
#         # xm = torch.linspace(0, 511, 512).view(
#         #     1, -1, 1).expand(1, 512, 1024)
#         # ym= torch.linspace(0, 1023, 1024).view(
#         #     1, 1, -1).expand(1, 512, 1024)
#         xym = torch.cat((xm, ym), 0)
#         self.register_buffer("xym", xym)
#
#         #cross entropy
#         self.criterion = nn.CrossEntropyLoss(weight=weight,
#                                              ignore_index=ignore_label,
#                                              reduction='none')
#
#     def bilinear_interpolate_torch(self,im, x, y):
#         # https: // gist.github.com / peteflorence / a1da2c759ca1ac2b74af9a83f69ce20e
#
#         x0 = torch.floor(x).type(dtype_long)
#         x1 = x0 + 1
#
#         y0 = torch.floor(y).type(dtype_long)
#         y1 = y0 + 1
#
#         x0 = torch.clamp(x0, 0, im.shape[1] - 1)
#         x1 = torch.clamp(x1, 0, im.shape[1] - 1)
#         y0 = torch.clamp(y0, 0, im.shape[0] - 1)
#         y1 = torch.clamp(y1, 0, im.shape[0] - 1)
#
#         Ia = im[y0, x0][0]
#         Ib = im[y1, x0][0]
#         Ic = im[y0, x1][0]
#         Id = im[y1, x1][0]
#
#         wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
#         wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
#         wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
#         wd = (x - x0.type(dtype)) * (y - y0.type(dtype))
#
#         return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
#             torch.t(Id) * wd)
#
#
#     def forward(self,s_i,target,w_s_i_2=1, w_s_s=1, w_s_f=1, w_f=1, **kwargs):
#         # print('s_i',s_i.size()) #--> (batch,19,512,1024)
#         batch_size,ph, pw = s_i.size(0), s_i.size(2), s_i.size(3) #batch size to check
#         h, w = target.size(1), target.size(2)
#         # print(s_i)
#         # print(h,w)
#         # print('s_i',s_i)
#         if ph != h or pw != w:
#             s_i = F.upsample(input=s_i, size=(h, w), mode='bilinear')
#         # print('s_i after',s_i)
#         xym_s = self.xym[:, 0:h, 0:w].contiguous() # 2 x h x w
#         tmp_target = target.clone() # batch x h x w
#         # print('tmp_target',tmp_target.size())
#         tmp_target[tmp_target == self.ignore_label] = 0 #ground truth
#         loss=0
#         # print('batchh size',batch_size) #2
#         # print('s_i',s_i.size()) # (2,3,512,1024)
#         # print('xym',xym_s.size())
#         #
#         # print('x',xym_s[0])
#         # print('y',xym_s[1])
#         # pred = F.softmax(s_i, dim=1)
#         # print(pred.size())
#         # gia kaue eikona
#         for b in range(0,batch_size):
#             spatial_emb = s_i[b, 0:2] + xym_s  # 2 x h x w  -> seed pixel
#             s_i_2=s_i[b,0:2]
#             # print('spatial_emb',spatial_emb.size())
#             f_map = torch.sigmoid(s_i[b, 2]) # 1 x h x w
#             # confidence map loss
#             print('0 before ',spatial_emb[0])
#             spatial_emb[0] = h * spatial_emb[0]
#             spatial_emb[1] = w * spatial_emb[1]
#             # print('0',spatial_emb[0])
#             # print('1',spatial_emb[1])
#             # print('f_map',f_map.size())
#             # PREPEI NA GINEI ALLAGI EDOO
#             s_s=torch.ones(tmp_target[b].size()) # prediction at seed location
#             # s_i_unsqueeze=torch.unsqueeze(torch.FloatTensor(s_i).type(dtype),2)
#             s_f = torch.ones(tmp_target[b].size()) #weighted prediction at p
#             for i in range(h):
#                 for j in range(w):
#                     x_cord = spatial_emb[0][i][j]
#                     y_cord = spatial_emb[1][i][j]
#                     if(x_cord < 0):
#                         x_cord = torch.tensor(0)
#                     elif(x_cord > (h-1)):
#                         x_cord = torch.tensor(h)
#                     if(y_cord<0):
#                         y_cord = torch.tensor(0)
#                     elif(y_cord>(w-1)):
#                         y_cord = torch.tensor(w)
#                     x_cord = torch.FloatTensor([x_cord])
#                     y_cord = torch.FloatTensor([y_cord])
#                     s_s[i][j] =self.bilinear_interpolate_torch(s_i_2,x_cord,y_cord)
#                     s_f[i][j]= (1-f_map[i][j])*s_i_2[b][i][j] + f_map[i][j]*s_s[i][j]
#             print(s_s)
#             print(s_s.size())
#             # print('0',torch.round( spatial_emb[0]))
#             # print('1',torch.round(spatial_emb[1]))
#             # indicator function
#             # print(spatial_emb[0].size()) #(512,1024)
#             print('tmp_target[b]', tmp_target[b])
#             tmp_target_seed = torch.ones(tmp_target[b].size())
#             for i in range(h):
#                 for j in range(w):
#                     tmp_target_seed[i][j] = tmp_target[b][spatial_emb[0][i][j]][spatial_emb[1[i][j]]]
#             print('tmp_target[b]',tmp_target[b])
#             print('tmp_target_seed',tmp_target_seed)
#             if tmp_target[b] != tmp_target_seed:
#                 f_loss-=torch.log(1-f_map)
#             else:
#                 f_loss+=torch.log(f_map)
#             # cross entropy losses for seed and confidence
#             loss_s_i_2 = self.criterion(s_i_2, target).contiguous().view(-1)
#             loss_s_s = self.criterion(s_s,target).contiguous().view(-1)
#             loss_s_f = self.criterion(s_f,target).contiguous().view(-1)
#             loss += w_s_i_2*loss_s_i_2 + w_s_s * loss_s_s + w_s_f * loss_s_f + w_f * f_loss
#         # loss = loss / (b+1)
#         return loss



