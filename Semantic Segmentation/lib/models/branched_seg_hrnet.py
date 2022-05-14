# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True,branch=False):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)

        self.branch = branch
        self.fuse_layers = self._make_fuse_layers(branch)
        # self.fuse_layers_branched = self._make_fuse_layers()

        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self,branch):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        if self.branch:
            fuse_layers2 = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            if (self.branch):
                fuse_layer2 = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    if (self.branch):
                        fuse_layer2.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j],
                                      num_inchannels[i],
                                      1,
                                      1,
                                      0,
                                      bias=False),
                            BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                    if (self.branch):
                        fuse_layer2.append(None)
                else:
                    conv3x3s = []
                    if (self.branch):
                        conv3x3s2 = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                            if (self.branch):
                                conv3x3s2.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j],
                                              num_outchannels_conv3x3,
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3,
                                                momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                            if(self.branch):
                                conv3x3s2.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j],
                                              num_outchannels_conv3x3,
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3,
                                                momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))  # etsi ua kano to fusion sta seeds
                    if(self.branch):
                        fuse_layer2.append(nn.Sequential(*conv3x3s2))
            fuse_layers.append(nn.ModuleList(fuse_layer))
            if (self.branch):
                fuse_layers2.append(nn.ModuleList(fuse_layer2))
        if(self.branch):
            return nn.ModuleList([nn.ModuleList(fuse_layers) ,nn.ModuleList(fuse_layers2)]) #thelei fuse_layers2
        else:
            return nn.ModuleList(fuse_layers)



    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        # print(self.num_branches)
        x_fuse = []

        if (self.branch):
            # t1 = self.fuse_layers[0:4]
            # # print(t1)
            # t2 = self.fuse_layers[4:8]
            # print(t1)
            # print(t2)
            # print(type(t1))
            # print(type(t2))
            # print(type(self.fuse_layers))
            # print(t1)
            # print(t1)
            x2_fuse=[] #for the extra layer
            for i in range(len(self.fuse_layers[0])):  # i ranges from 0 to 7
                y = x[0] if i == 0 else self.fuse_layers[0][i][0](x[0])
                y1= x[0] if i == 0 else self.fuse_layers[1][i][0](x[0])
                for j in range(1, self.num_branches):  # j ranges from 0 to 3
                    if i == j:
                        y = y + x[j]
                        y1 = y1 + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        # print(type(t1[i][j](x[j])))
                        y = y + F.interpolate(
                            self.fuse_layers[0][i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                        y1 = y1 + F.interpolate(
                            self.fuse_layers[1][i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                    else:
                        y = y + self.fuse_layers[0][i][j](x[j])
                        y1 = y1 + self.fuse_layers[1][i][j](x[j])
                x_fuse.append(self.relu(y))
                x2_fuse.append(self.relu(y1))
            x_fuse=x_fuse+x2_fuse
        else:
            # print('FUSE',self.fuse_layers)
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        y = y + F.interpolate(
                            self.fuse_layers[i][j](x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True,branch=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        # print('last inp channels',last_inp_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.offset_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=3,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_SEED == 3 else 0)
        )
        # coordinate map

        # x ->  [ 0 ... 1023]
        #            .
        #            .
        #       [ 0 ... 1023]

        # y ->  [ 0 ... 0]
        #            .
        #            .
        #      [511 ... 511]
        xm = torch.linspace(0, 1023, 1024).view(
            1, 1, -1).expand(1, 512, 1024)     # 1 x 512 x 1024
        ym = torch.linspace(0, 511, 512).view(
            1, -1, 1).expand(1, 512, 1024)
        xym = torch.cat((xm, ym), 0)          # 1 x 512 x 1024
        self.register_buffer("xym", xym)

        # xm = torch.linspace(0, 2, 1024).view(
        #     1, 1, -1).expand(1, 512, 1024)
        # ym = torch.linspace(0, 1, 512).view(
        #     1, -1, 1).expand(1, 512, 1024)
        # xym = torch.cat((xm, ym), 0)
        # self.register_buffer("xym", xym)


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True,branch=False):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output,branch)
            )
            num_inchannels = modules[-1].get_num_inchannels()
        # print('num_inchannels',num_inchannels)
        # print('lenght is',len(modules))
        # if(branch):
        #     print(nn.Sequential(*modules))
        return nn.Sequential(*modules), num_inchannels

    def bilinear_interpolate_torch(self,im, x, y):

        x0 = torch.floor(x).type(dtype_long)
        x1 = x0 + 1

        y0 = torch.floor(y).type(dtype_long)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[2] - 1)
        x1 = torch.clamp(x1, 0, im.shape[2] - 1)
        y0 = torch.clamp(y0, 0, im.shape[1] - 1)
        y1 = torch.clamp(y1, 0, im.shape[1] - 1)


        Ia = im[:,y0, x0]
        Ib = im[:,y1, x0]
        Ic = im[:,y0, x1]
        Id = im[:,y1, x1]

        wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
        wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
        wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
        wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

        return  Ia * wa + Ib * wb + Ic * wc + Id * wd
        # return torch.nn.Parameter(torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        #     torch.t(Id) * wd))

    def seed_prediction(self,s_s,s_i,x_cords ,y_cords):
        for b in range(0,s_i.size(0)):
            for i in range(0, s_i.size(1)):
                s_s[b, i] = self.bilinear_interpolate_torch(s_i[b, i].unsqueeze(0), x_cords[b].squeeze(0), y_cords[b].squeeze(0))

        # slow solution
        # for y in range(h):
        #     for x in range(w):
        #         x_cord=x_cords[:,y,x]  # batch
        #         y_cord=y_cords[:,y,x]  # batch
        #
        #         # Grid Limits
        #         mask_1 = x_cord < 0
        #         mask_2 = x_cord > (w-1)
        #         mask_3 = y_cord < 0
        #         mask_4 = y_cord > (h-1)
        #
        #         if(True in mask_1):
        #             x_cord[mask_1] = 0
        #         elif(True in mask_2):
        #             x_cord[mask_2] = w-1
        #         if(True in mask_3):
        #             y_cord[mask_3] = 0
        #         elif(True in mask_4):
        #             y_cord[mask_4] = h-1
        #
        #         # s_i[:]=torch.unsqueeze(torch.FloatTensor(s_i[:]).type(dtype),2)
        #         x_cord = torch.FloatTensor([x_cord]).type(dtype)
        #         y_cord = torch.FloatTensor([y_cord]).type(dtype)
        #
        #         for i in range(0,s_i.size(1)): # for each of 19 classes
        #             # print(s_i[:,i].size())
        #             s_s[:,i,y,x]= self.bilinear_interpolate_torch(s_i[:,i].squeeze(0),x_cord,y_cord)
        return s_s




    def forward(self, x):
        # print('0',x)
        x = self.conv1(x)
        # print('1',x)
        x = self.bn1(x)
        # print('2', x)
        x = self.relu(x)
        # print('3', x)
        x = self.conv2(x)
        # print('4', x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        # print('one check here',x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        # print('print layer 3', y_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        # print('last stage here',x)
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        # Extra layer
        x4_h,x4_w=x[4].size(2), x[4].size(3)
        x5 = F.upsample(x[5], size=(x4_h, x4_w), mode='bilinear')
        x6 = F.upsample(x[6], size=(x4_h, x4_w), mode='bilinear')
        x7 = F.upsample(x[7], size=(x4_h, x4_w), mode='bilinear')

        x_1 = torch.cat([x[0], x1, x2, x3], 1) # batch x 720 x 128 x 256
        x_2 = torch.cat([x[4], x5, x6, x7], 1)
        # x = torch.cat([x[0],x[4],x1,x5,x2,x6,x3,x7],1)

        # initial prediciton of the network at p
        scores = self.last_layer(x_1) # batch x 19 x h x w

        # offset vector prediction and confidence map
        # x_2 = x_2.clone()

        o_f = self.offset_layer(x_2) # batch x 3 x h x w
        # o_f = o_f.clone()
        # print('Scores',scores)
        # print('CHECK HERE',o_f)


        #mapping for confidence maps
        # f=(o_f[:, 2] + 1)/2 # batch x h x w
        f = self.sigmoid(o_f[:, 2]) # batch x h x w
        # print('CONFIDENCE',f)
        # f = o_f[:, 2]
        o_f[:, 2] = f
        f = f.unsqueeze(1)

        # scaling
        o_f[:,0:2] = self.tanh(o_f[:,0:2]) * 100 # 100 pixels the biggest distance
        # print('OFFSET',o_f[:,0:2])


        #predictions
        s_i = F.softmax(scores,dim=1) #logits to predictions through softmax
        s_s = torch.ones(s_i.size())  # batch x 19 x h x w
        # s_f = torch.ones(s_i.size())  # batch x 19 x h x w

        h, w= s_i.size(2),s_i.size(3)

        xym_s = self.xym[:, 0:h, 0:w].contiguous()  # 2 x h x w
        spatial_pix=o_f[:,0:2] + xym_s # batch x 2 x h x w


        x_cords = spatial_pix[:,0] # batch x h x w
        y_cords = spatial_pix[:,1] # batch x h x w

        x_cords = torch.clamp(x_cords, 0, w - 1)
        y_cords = torch.clamp(y_cords, 0, h - 1)

        s_s = self.seed_prediction(s_s,s_i,x_cords,y_cords)
        s_s=s_s.type(dtype)

        # print('s_s',s_s.size())
        # print('f',f.size())
        # print('s_i',s_i.size())
        s_f = (1 - f) * s_i + f * s_s # batch x 19 x h x w
        s_f = s_f.type(dtype)  # <class  Torch Tensor >

        return s_i,o_f,s_s,s_f

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model