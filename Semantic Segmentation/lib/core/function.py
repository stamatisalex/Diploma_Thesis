# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
# import wandb
import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank

from PIL import Image

from .flow_vis import flow_to_color, flow_uv_to_colors, make_colorwheel

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device,off_vis=True,sv_dir=''):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    # ave_seed_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, instances,name,_ = batch # ua xreiastei to instances
        # print(instances)
        # print(instances.size())
        # print(instances[0].size())
        # print(w)
        # print(instances)
        # print(name)
        # print('IMAGES',images.size())


        images = images.to(device)
        labels = labels.long().to(device)
        # With o_f
        # losses, _,o_f  = model(images, labels) #inputs, labels

        # Without o_f
        losses, _, o_f = model(images, labels)  # inputs, labels
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        # ave_seed_loss.update(reduced_seed_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        for n, p in model.named_parameters():
            # print('{0} and {1}'.format(n, p))
            if p.grad is None:
                print(f'{n} has no grad')

###########################################################################
        # Offset Visualization  uncomment this
        # print("o_f",o_f.size())  #128x256
        # print("labels",labels.size()) #512x1024
        if off_vis:
            sv_path = os.path.join(sv_dir, 'offset_results')
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            size = labels.size()
            o_f = F.upsample(input=o_f, size=(
                        size[-2], size[-1]), mode='bilinear')
            # print("o_f",o_f.size())
            o_f = o_f.cpu().detach().numpy()
            for i in range(o_f.shape[0]):
                flow_color = flow_to_color(np.moveaxis(o_f[i,0:2], 0, -1), convert_to_bgr=False)
                flow_color = Image.fromarray(flow_color)

                # wandb.log({"offset_visualization": [wandb.Image(flow_color,caption= name[i] + '.png')]})
                flow_color.save(os.path.join(sv_path, name[i] + '.png'))
#####################################################################################
        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            # print_seed_loss = ave_seed_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), lr, print_loss)
            logging.info(msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ , name = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            # With o_f
            losses, pred,_ = model(image, label)
            # Without o_f
            # losses, pred = model(image, label)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            for n, p in model.named_parameters():
                if p.grad is None:
                    print(f'{n} has no grad')

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array
    

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name,_ = batch
            size = label.size()
            # print(1)
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            # print(6)
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
