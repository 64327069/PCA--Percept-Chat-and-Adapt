# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import sys
import time
from random import random
from torch.optim import lr_scheduler

import pandas as pd
from torch import optim

import clip
import numpy as np
import torch
import torch.nn as nn
from datasets.ARID_mod import ARID_MOD
from torch.utils.data import DataLoader
import video_transforms
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from modules.CLIP_domain_model import CLIP_domain_model
from modules.PromptLearner import PromptLearner
from modules.GptPromptLearner import GptPromptLearner
from modules.Fine_grained_prompt import Fine_grained_prompt
from utils.KLLoss import KLLoss
from test import validate, clip_validate, validate_multicls, validateF1_multicls, fg_clip_validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *
from utils.loss import CrossEn, BCEloss, MLT_loss



def set_seed_logger(seed):
    global logger
    # predefining random initial seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # True 的话，将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    torch.backends.cudnn.deterministic = True  # 确保训练初始值固定

    return args

# get the model, need to be modified into the config
def get_feat(config):
    # for split i 
    from models.dark_light_feat import dark_light_feat
    model_path = config.pretrain
    model = dark_light_feat(num_classes=11)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    print("load model success {}".format(model_path))
    
    return model
    


# 进入,一定不要出现两个Datapararallel
def train_epoch(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model, lmodel = model
    # switch to train mode
    model.train()
    lmodel.eval()
    input_size = config.data.input_size
    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter = 0
    # return
    for i, (inputs, inputs_light, targets, text_feat) in tqdm(enumerate(train_loader)):
        
        inputs = inputs.view(-1, config.data.num_segments, 3, input_size, input_size).transpose(1, 2)
        inputs_light = inputs_light.view(-1, config.data.num_segments, 3, input_size, input_size).transpose(1, 2)
        inputs = inputs.to('cuda', non_blocking=True)
        text_feat = text_feat.to('cuda', non_blocking=True)
        inputs_light = inputs_light.to('cuda', non_blocking=True)
        targets = targets.to('cuda', non_blocking=True)
        _, light_output = lmodel(inputs_light)
        output = model(inputs, light_output, text_feat)

        prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
        acc_mini_batch += prec1.item()
        acc_mini_batch_top3 += prec5.item()

        lossClassification = criterion(output, targets)
        lossClassification = lossClassification / args.iter_size

        totalLoss = lossClassification
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter += output.size(0)
        if (i + 1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch / args.iter_size, totalSamplePerIter)
            top5.update(acc_mini_batch_top3 / args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0
            # scheduler.step()

        if (i + 1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' % (i, batch_time.avg, lossesClassification.avg))

    print(
        'train * Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
        .format(epoch=epoch, top1=top1, top5=top5, lossClassification=lossesClassification))


def validate_epoch(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model, lmodel = model
    model.eval()
    lmodel.eval()
    input_size = config.data.input_size
    length = config.data.num_segments
    end = time.time()
    with torch.no_grad():
        for i, (inputs, inputs_light, targets, text_feat) in tqdm(enumerate(val_loader)):
            # break
            
            inputs = inputs.view(-1, length, 3, input_size, input_size).transpose(1, 2)
            inputs_light = inputs_light.view(-1, length, 3, input_size, input_size).transpose(1, 2)
            text_feat = text_feat.cuda()
            inputs = inputs.cuda()
            inputs_light = inputs_light.cuda()
            targets = targets.cuda()
            # compute output
            _, light_output = lmodel(inputs_light)
            output = model(inputs, light_output, text_feat)
            lossClassification = criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))

            lossesClassification.update(lossClassification.data.item(), output.size(0))

            top1.update(prec1.item(), output.size(0))
            top5.update(prec5.item(), output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(
            'validate * * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
            .format(top1=top1, top5=top5, lossClassification=lossesClassification))

    return top1.avg, top5.avg, lossesClassification.avg


def save_checkpoint(state, resume_path):
    cur_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def resume_model(model, optimizer, resume_path):
    checkpoint = torch.load(resume_path)
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    print("the best _prec1 is {}".format(checkpoint['best_prec1']))
    return model, optimizer, start_epoch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)
    set_seed_logger(config.seed)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train_all_feat.py', working_dir)
    shutil.copy('models/dark_light_mod1.py', working_dir)

    args.iter_size = 1
    args.epochs = config.solver.epochs
    args.print_freq = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    if config.network.arch == "R(2+1)D":
        from models.dark_light_mod1 import dark_light
        
    model = dark_light(num_classes=11)
    model = torch.nn.DataParallel(model).cuda()
    light_model = get_feat(config)
    light_model = torch.nn.DataParallel(light_model).cuda()
    length = config.data.num_segments
    input_size = config.data.input_size
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [0.485, 0.456, 0.406] * 1 * length
    clip_std = [0.229, 0.224, 0.225] * 1 * length

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    transform_train = video_transforms.Compose([
        video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        normalize,
    ])

    transform_val = video_transforms.Compose([
        video_transforms.CenterCrop((input_size)),
        video_transforms.ToTensor(),
        normalize,
    ])

    width = 170
    height = 128
    train_dataset = ARID_MOD(root=config.data.datapath,
                            modality="rgb",
                            source=config.data.train_list,
                            phase="train",
                            is_color=True,
                            new_length=length,
                            new_width=width,
                            new_height=height,
                            video_transform=transform_train,
                            num_segments=1,
                            gamma=1.8)

    val_dataset = ARID_MOD(root=config.data.datapath,
                            modality="rgb",
                            source=config.data.val_list,
                            phase="val",
                            is_color=True,
                            new_length=length,
                            new_width=width,
                            new_height=height,
                            video_transform=transform_val,
                            num_segments=1,
                            gamma=1.8)


    print('{} samples found, {} train data and {} test data.'.format(len(val_dataset) + len(train_dataset),
                                                                     len(train_dataset),
                                                                     len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size, shuffle=True,
        num_workers=config.data.workers, pin_memory=True)
    print(train_loader)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.workers, pin_memory=True)    

    optimizer = optim.AdamW(model.module.features.parameters(),
                            betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
                            weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    start_epoch = 0
    if config.resume is not None:
        model, optimizer, start_epoch = resume_model(model, optimizer, config.resume)
        
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    best_prec1 = 0.0
    # for k, v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
    # for k, v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, args.epochs):
        train_epoch(train_loader, (model, light_model), criterion, optimizer, epoch, config)
        prec1, prec5, lossClassification = validate_epoch(val_loader, (model, light_model), criterion, config)
        scheduler.step(lossClassification)
        if prec1 >= best_prec1:
            best_prec1 = prec1
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': config.network.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                },  working_dir)
            print("best prec is {}".format(best_prec1))



    




if __name__ == '__main__':
    main()
