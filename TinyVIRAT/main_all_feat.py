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
import torch.distributed as dist
from torch.backends import cudnn
from torch.optim import SGD, Adam, lr_scheduler, AdamW
from einops import rearrange
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

def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                model_type,
                current_lr,
                epoch_logger,
                tb_writer=None,
                model_sr = None,
                distributed=False):
    print('train at epoch {}'.format(epoch))
    # return 
    model.train()
    # return 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #accuracies = AverageMeter()
    precision_all = []
    recall_all = []
    f1_all = []

    end_time = time.time()
    # assert model_sr is not none
    assert model_sr is not None
    model_sr.eval()
    # return 
    for i, (inputs, sr_clips, targets, _, bert_feat) in enumerate(data_loader):
        # break
        data_time.update(time.time() - end_time)
        inputs = inputs.to(device, non_blocking=True)
        sr_clips = sr_clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bert_feat = bert_feat.to(device, non_blocking=True)
        _, clip_feat = model_sr(sr_clips)
        outputs = model(inputs, clip_feat, bert_feat)
        if model_type == 'i3d':
            outputs = torch.max(outputs, dim=2)[0]
        loss = criterion(outputs, targets) 
        precision, recall, f1 = calculate_precision_recall_f1(outputs, targets, 26, 0.5)
        losses.update(loss.item(), inputs.size(0))
        precision_all.append(precision)
        recall_all.append(recall)
        f1_all.append(f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        # torch.cuda.empty_cache()

        print('Epoch: [{0}][{1}/{2}]\t'
              'precision: {3:.3f}, recall: {4:.3f}, f1:{5:.3f}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         precision, recall, f1,
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses))

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'precision': np.average(np.array(precision_all)),
            'recall': np.average(np.array(recall_all)),
            'f1': np.average(np.array(f1_all)),
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/f1', np.average(np.array(f1_all)), epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
    torch.cuda.empty_cache()


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              model_type,
              logger,
              wrong_video_log,
              tb_writer=None,
              model_sr = None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #accuracies = AverageMeter()
    precision_all = []
    recall_all = []
    f1_all = []
    # wrong_video_path = []
    # import pdb
    # pdb.set_trace()
    
    assert model_sr is not None
    
    end_time = time.time()
    # anno_dict = get_dict()
    with torch.no_grad():
        for i, (inputs, sr_clips, targets, video_paths, bert_feat) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True)
            #TODO: if use batchformer change here
            bert_feat = bert_feat.to(device, non_blocking=True)
            _, clip_feat = model_sr(sr_clips)
            # last_feat = last_feat.to(device, non_blocking=True)
            
            outputs = model(inputs, clip_feat, bert_feat)
            # outputs = model(inputs, bert_feat, clip_feat, last_feat)
            
            #if model_type == 'i3d':
             #   outputs = torch.max(outputs, dim=2)[0]
            #outputs = torch.mean(outputs, dim=0, keepdims = True)[0]
            #outputs = outputs.reshape(-1, 26)
            
            loss = criterion(outputs, targets)
            # outputs = post_process(outputs, video_paths, anno_dict, targets)
            
            #acc = calculate_accuracy(outputs, targets)
           
            precision, recall, f1 = calculate_precision_recall_f1(outputs, targets, 26, 0.5)
            # print wrong video: 
            
            #if f1 < 0.5:
                # import pdb
                # pdb.set_trace()
                #wrong_video_log.log({'video_path': video_paths, 'f1': f1, 'pred_label': outputs, 'truth': targets})
            
            losses.update(loss.item(), inputs.size(0))
            #accuracies.update(acc, inputs.size(0))
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
            
    '''
    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()
    '''
    f1_avg = np.average(np.array(f1_all))
    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'precision': np.average(np.array(precision_all)),
                                                        'recall': np.average(np.array(recall_all)),
                                                        'f1': f1_avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/f1', np.average(np.array(f1_all)), epoch)
    torch.cuda.empty_cache()

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
    # begin_epoch = 1
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        except:
            # pass
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


def get_train_utils(opt, cfg, model):
    
    # Get training data 
    # I think shuffle the train data is necessary
    train_data = TinyVirat(cfg, 'train', 1.0, num_frames=opt.num_frames, skip_frames=opt.skip_frames, input_size=opt.sample_size)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads, drop_last = True)
                                               #pin_memory=True,
                                               #sampler=train_sampler,
                                               #worker_init_fn=worker_init_fn)

    out_file_path = 'train_{}.log'.format(opt.model)
    train_logger = Logger(opt.result_path / out_file_path, ['epoch', 'loss', 'precision', 'recall', 'f1', 'lr'])
    
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    
    if opt.optimizer == 'sgd':
        optimizer = SGD(model.parameters(),
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        dampening=dampening,
                        weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = Adam(model.parameters(),
                            lr=opt.learning_rate,
                            weight_decay=0,
                            eps=1e-8)
    # lr need to be changed in this place??? 
    elif opt.optimizer == 'adamW':
        
        # para_text = model.module.transformer.resblocks[-1].text_block.parameters()
        # cros_text = model.module.transformer.resblocks[-1].cross_attn_text.parameters()
        # para_groups = [{'params': para_text}, {'params': cros_text}]
        
       
        # for name, param in model.module.named_parameters():
        #     if 'text_block' not in name and 'cross_attn_text' not in name and 'proj.0' not in name:
        #         param.requires_grad = False
        # for name, param in model.module.named_parameters():
        #     if 'addition_block'  in name:
        #         param.requires_grad = False
        # for name, param in model.module.named_parameters():
        #     if 'text_block' in name or 'cross_attn_text'  in name:
        #         param.requires_grad = False
        optimizer = AdamW(model.parameters(), lr = opt.learning_rate, weight_decay = opt.weight_decay)
        
        # for name, param in model.module.named_parameters():
        #     print(name, param.requires_grad)
        # my_scheduler = Scheduler(optimizer, xxxx)
    else:
        print("="*40)
        print("Invalid optimizer mode: ", opt.optimizer)
        print("Select [sgd, adam]")
        exit(0)
        
    assert opt.lr_scheduler in ['plateau', 'multistep', 'cosine']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    # opt.lr_scheduler = 'haha'
    # TODO:change the scheduler to cosine
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    # add this
    elif opt.lr_scheduler == 'cosine':
        # TODO: change here for epoch num
        scheduler = CosineAnnealingWarmRestarts(optimizer, opt.n_epochs, warmup = 1)
        print("I use coscine\n----------------------------------")
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[20, 40, 60], gamma=0.5)

    return (train_loader, train_logger, optimizer, scheduler)


def get_val_utils(opt, cfg):
    
    # Get validation data 
    val_data = TinyVirat(cfg, 'val', 1.0, num_frames=opt.num_frames, skip_frames=opt.skip_frames, input_size=opt.sample_size)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=int(opt.batch_size / 3),
                                             shuffle=False,
                                             num_workers=opt.n_threads)
                                             #pin_memory=True,
                                             #sampler=val_sampler,
                                             #worker_init_fn=worker_init_fn)
    
    out_file_path = 'val_{}.log'.format(opt.model)
    val_logger = Logger(opt.result_path / out_file_path, ['epoch', 'loss', 'precision', 'recall', 'f1'])
    
    return val_loader, val_logger


def get_inference_utils(opt):
    
    # Get inference data 
    #inference_data = #inference data loader
    inference_data = 1
    
    inference_loader = torch.utils.data.DataLoader(
                                        inference_data,
                                        batch_size=opt.inference_batch_size,
                                        shuffle=False,
                                        num_workers=opt.n_threads)
                                        #pin_memory=True,
                                        #worker_init_fn=worker_init_fn,
                                        #collate_fn=collate_fn)

    return inference_loader, inference_data.class_names
    

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
            
        # wrong video / epoch path
        if not os.path.exists(opt.result_path/'wrong_video'):
            os.makedirs(opt.result_path/'wrong_video')
        
    # import pdb 
    # pdb.set_trace()
    
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
    
    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)
    
    return opt

def generate_model_sr():
    path = '/data0/workplace/tiny_virat/results/cnn_clip_528_1SR/save_18.pth'
    model = cnn_clip_feat_b16(input_resolution = 224 ,pretrained_path = None, num_classes = 400) 
    model.proj = nn.Sequential(
    nn.LayerNorm(model.proj[2].in_features),
    nn.Dropout(0.8),
    nn.Linear(model.proj[2].in_features, 26))
    model = resume_model(path, 'cnn_clip-18', model)
    
    # path = '/data0/workplace/tiny_virat/weights/timesformer_save_best.pth'
    # model = timesformer()  
    # model.model.head = nn.Sequential(nn.Dropout(0.), nn.Linear(model.model.head.in_features, 26))
    # model = resume_model(path, 'timesformer-18', model)
    # path = '/data0/workplace/tiny_virat/results/SR_uniformer/save_best.pth'
    # model = uniformer_b_16()  
    # model.head = nn.Sequential(nn.Dropout(0.), nn.Linear(model.head.in_features, 26))
    # model = resume_model(path, 'uniformer-18', model)

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
    
    if not opt.no_train:
        (train_loader, train_logger, optimizer, scheduler) = get_train_utils(opt, cfg, model)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt, cfg)
        
    criterion = AsymmetricLoss().to(opt.device)
    
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None
        
    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        
        if not opt.no_train:
            current_lr = get_lr(optimizer)
           
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, opt.model, current_lr, train_logger,
                        tb_writer, model_sr = model_sr)

            if i % opt.checkpoint == 0:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)
        # 3 epoch one val
        torch.cuda.empty_cache()
        if not opt.no_val and i % 1 == 0:
            out_file_path = 'wrong_video/epoch_{}.log'.format(i)
            wrong_video_log = Logger(opt.result_path / out_file_path, ['video_path', 'pred_label', 'truth'])
            prev_val_loss1, f1_avg1 = val_epoch(i, val_loader, model, criterion,
                                      opt.device, opt.model, val_logger, wrong_video_log, tb_writer, model_sr = model_sr)
            if f1_avg1 > 0.813:
                prev_val_loss2, f1_avg2 = val_epoch(i, val_loader, model, criterion,
                                      opt.device, opt.model, val_logger, wrong_video_log, tb_writer, model_sr = model_sr)
            
                f1_avg = (f1_avg1 + f1_avg2) / 2
                prev_val_loss = (prev_val_loss1 + prev_val_loss2) / 2
                
            else:
                f1_avg = f1_avg1
                prev_val_loss = prev_val_loss1
                
            if f1_avg > best_f1:
                best_f1 = f1_avg
                save_file_path = opt.result_path / 'save_best.pth'
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)
        torch.cuda.empty_cache()
            

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)
        elif not opt.no_train and opt.lr_scheduler == 'cosine':
            scheduler.step()
        
    
    
if __name__ == '__main__':
    
    opt = get_opt()
    
    if not opt.no_cuda:
        cudnn.benchmark = True
    
    main_worker(opt)
    
    
    
    
