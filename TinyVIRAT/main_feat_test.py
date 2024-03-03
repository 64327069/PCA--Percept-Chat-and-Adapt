import torch
import time
import os
import sys
import numpy as np
import json 
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# 这个好像没有用到
from torch.backends import cudnn
from torch.optim import SGD, Adam, lr_scheduler, AdamW
from einops import rearrange
# cos
from models.clip_feat_extract import cnn_clip_feat_b16
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from opts import parse_opts
from model_wo_feat import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
# from utils import AverageMeter, calculate_accuracy, calculate_precision_recall_f1, AdamW
from utils import AverageMeter, calculate_accuracy, calculate_precision_recall_f1

from utils import Logger, worker_init_fn, get_lr, CosineAnnealingWarmRestarts
from configuration import build_config
# clip
from dataloader_withfeat import Normalize, TinyVirat
import warnings
from losses import AsymmetricLoss, AsymmetricLossOptimized
warnings.filterwarnings("ignore")

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              model_type,
              logger,
              tb_writer=None,
              model_sr = None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precision_all = []
    recall_all = []
    f1_all = []
    
    assert model_sr is not None
    
    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, sr_clips, targets, video_paths, bert_feat) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True)
            bert_feat = bert_feat.to(device, non_blocking=True)
            _, clip_feat = model_sr(sr_clips)
            
            outputs = model(inputs,clip_feat, bert_feat)            
            loss = criterion(outputs, targets)
            precision, recall, f1 = calculate_precision_recall_f1(outputs, targets, 26, 0.5)
            
            losses.update(loss.item(), inputs.size(0))
            precision_all.append(precision)
            recall_all.append(recall)
            f1_all.append(f1)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'precision: {3:.3f}, recall: {4:.3f}, f1:{5:.3f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      precision, recall, f1,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))

    f1_avg = np.average(np.array(f1_all))
    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'precision': np.average(np.array(precision_all)),
                                                        'recall': np.average(np.array(recall_all)),
                                                        'f1': f1_avg})

    return losses.avg, f1_avg


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    return model
    

def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        except:
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint['optimizer']['param_groups'][0]['lr'] 

        

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)
          
    
def get_mean_std(value_scale, dataset):
    assert dataset in ['TinyVirat', 'activitynet', 'kinetics', '0.5']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'TinyVirat':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]    

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std


def get_val_utils(opt, cfg):
    val_data = TinyVirat(cfg, 'val', 1.0, num_frames=opt.num_frames, skip_frames=opt.skip_frames, input_size=opt.sample_size)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=int(opt.batch_size / 3),
                                             shuffle=False,
                                             num_workers=opt.n_threads)
    out_file_path = 'val_{}.log'.format(opt.model)
    val_logger = Logger(opt.result_path / out_file_path, ['epoch', 'loss', 'precision', 'recall', 'f1'])
    return val_loader, val_logger

    

def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)
    

def get_opt():
    opt = parse_opts()
    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path
    
    if opt.sub_path is None:
        opt.sub_path = opt.model
        
    if opt.sub_path is not None:
        opt.result_path = opt.result_path / opt.sub_path
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
                
    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]
    opt.no_cuda = True if not torch.cuda.is_available() else False 
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    print(opt)
    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)
    
    return opt

def generate_model_sr():
    path = '/data0/workplace/tiny_virat/results/cnn_clip_5281SR/new_sr.pth'
    model = cnn_clip_feat_b16(input_resolution = 224 ,pretrained_path = None, num_classes = 400) 
    model.proj = nn.Sequential(
    nn.LayerNorm(model.proj[2].in_features),
    nn.Dropout(0.5),
    nn.Linear(model.proj[2].in_features, 26))
    model = resume_model(path, 'cnn_clip-18', model)
    model = model.to('cuda')
    return model


def main_worker(opt):
    best_f1 = 0.0
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    cfg = build_config("TinyVirat")
    model = generate_model(opt)
    model_sr = generate_model_sr()
    
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path and 'clip' not in opt.model   :    
            model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    print(opt.device)
    print("load model done")
    model = make_data_parallel(model, opt.distributed, opt.device)
    model_sr = make_data_parallel(model_sr, opt.distributed, opt.device)
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt, cfg)
        
    criterion = AsymmetricLoss().to(opt.device)
    tb_writer = None
    _, f1_avg = val_epoch(0, val_loader, model, criterion, opt.device, opt.model, val_logger, tb_writer, model_sr = model_sr)
    print('f1_avg1 is {}'.format(f1_avg))
    torch.cuda.empty_cache()
            
if __name__ == '__main__':
    
    opt = get_opt()
    if not opt.no_cuda:
        cudnn.benchmark = True
    main_worker(opt)