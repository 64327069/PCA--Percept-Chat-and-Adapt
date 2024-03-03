# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
from einops import rearrange
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from modules.finegrained_model import finegrained_model,Single_Mode_Head
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
from utils.eval_mAP import MAP

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def validate(epoch, val_loader, classes, device, model, fusion_model,fg_model, sam_model, config, num_text_aug):
    model.eval()
    fusion_model.eval()
    sam_model.eval()
    fg_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            cls_scores = fg_model(image_features,text_features)
            cls_scores = cls_scores.softmax(dim=-1)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ text_features.T)
            # similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            # similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = cls_scores.topk(1, dim=-1)
            values_5, indices_5 = cls_scores.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    return top1

def validate_multicls_all_feat(epoch, val_loader, classes, device, model_image, fusion_model,fg_model,sam_model, config):
    model_image.eval()
    fusion_model.eval()
    sam_model.eval()
    fg_model.eval()
    pre_list = []
    label_list = []

    with torch.no_grad():
        if config.network.multi_mode:
            text_inputs = classes.to(device)
            text_features = model_image.encode_text(text_inputs)
        for iii, (images, sam_images, class_id, text_feat) in enumerate(tqdm(val_loader)):
            
            if config.network.type == 'cnn_clip' or config.network.type == 'timesformer' or config.network.type == 'uniformer':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                images = rearrange(images, 'b t c h w -> b c t h w')                
                image_embedding = model_image(images)
                
            elif config.network.type == 'clip':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                sam_images = sam_images.view((-1,config.data.num_segments,3)+ sam_images.size()[-2:])
                # images = rearrange(images, 'b c t h w -> b t c h w')
                
                b,t,c,h,w = images.size()
            

                images= images.to(device).view(-1,c,h,w ) #
                sam_images = sam_images.to(device).view(-1,c,h,w ) #
                # (256, 3, 224, 224) -> (256, 512)
                _, sam_feat = sam_model(sam_images)
                text_feat = text_feat.to(device)
                # image_embedding = model_image(images, sam_feat)
                image_embedding = model_image(images, sam_feat, text_feat)
                # (32,8,512)
                image_embedding = image_embedding.view(b,t,-1)
                
            class_id = [x.tolist() for x in class_id]
            class_id =torch.tensor(class_id).t()
            # class_id.tolist()
            # image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_embedding)
            if config.network.multi_mode:
              cls_scores = fg_model(image_features,text_features)
            else:
              cls_scores = fg_model(image_features)
            cls_scores = torch.sigmoid(cls_scores)
            pre_list += cls_scores.tolist()
            label_list += class_id.tolist()
            
    mAP = MAP(label_list,pre_list,config.data.num_classes) * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: mAP: {}'.format(epoch, config.solver.epochs, mAP))
    return mAP

def validate_multicls_all_feat_other(epoch, val_loader, classes, device, model_image, sam_model, config):
    model_image.eval()
    sam_model.eval()
    pre_list = []
    label_list = []

    with torch.no_grad():

        for iii, (images, sam_images, class_id, text_feat) in enumerate(tqdm(val_loader)):
            
            if config.network.type == 'cnn_clip' or config.network.type == 'timesformer' or config.network.type == 'uniformer' or config.network.type == 'r2p1d':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                sam_images = sam_images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                text_feat = text_feat.to(device)
                images = rearrange(images, 'b t c h w -> b c t h w')  
                sam_images = rearrange(sam_images, 'b t c h w -> b c t h w')  
                _, img_feat = sam_model(sam_images)
                cls_scores = model_image(images, img_feat, text_feat)
                
            elif config.network.type == 'clip':
                raise ValueError('config.image_clip_mode must be ')
                
            class_id = [x.tolist() for x in class_id]
            class_id =torch.tensor(class_id).t()
            # class_id.tolist()
            # image_features = model.encode_image(image_input).view(b, t, -1)
            cls_scores = torch.sigmoid(cls_scores)
            pre_list += cls_scores.tolist()
            label_list += class_id.tolist()
            
    mAP = MAP(label_list,pre_list,config.data.num_classes) * 100

    print('Epoch: [{}/{}]: mAP: {}'.format(epoch, config.solver.epochs, mAP))
    return mAP


def validate_multicls(epoch, val_loader, classes, device, model_image, fusion_model,fg_model,sam_model, config):
    model_image.eval()
    fusion_model.eval()
    sam_model.eval()
    fg_model.eval()
    pre_list = []
    label_list = []

    with torch.no_grad():
        if config.network.multi_mode:
            text_inputs = classes.to(device)
            text_features = model_image.encode_text(text_inputs)
        for iii, (images, sam_images, class_id, path) in enumerate(tqdm(val_loader)):
            import ipdb; ipdb.set_trace()
            if config.network.type == 'cnn_clip' or config.network.type == 'timesformer' or config.network.type == 'uniformer':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                images = rearrange(images, 'b t c h w -> b c t h w')                
                image_embedding = model_image(images)
                
            elif config.network.type == 'clip':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                sam_images = sam_images.view((-1,config.data.num_segments,3)+ sam_images.size()[-2:])
                # images = rearrange(images, 'b c t h w -> b t c h w')
                
                b,t,c,h,w = images.size()
            
                images= images.to(device).view(-1,c,h,w ) #
                sam_images = sam_images.to(device).view(-1,c,h,w ) #
                # (256, 3, 224, 224) -> (256, 512)
                _, sam_feat = sam_model(sam_images)
                # image_embedding = model_image(images, sam_feat)
                image_embedding = model_image(images, sam_feat)
                # (32,8,512)
                image_embedding = image_embedding.view(b,t,-1)
                
            class_id = [x.tolist() for x in class_id]
            class_id =torch.tensor(class_id).t()
            # class_id.tolist()
            # image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_embedding)
            if config.network.multi_mode:
              cls_scores = fg_model(image_features,text_features)
            else:
              cls_scores = fg_model(image_features)
            cls_scores = torch.sigmoid(cls_scores)
            pre_list += cls_scores.tolist()
            label_list += class_id.tolist()
            
    mAP = MAP(label_list,pre_list,config.data.num_classes) * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: mAP: {}'.format(epoch, config.solver.epochs, mAP))
    return mAP

def validate_multicls_other_text(epoch, val_loader, classes, device, model_image, sam_model, config):
    model_image.eval()

    sam_model.eval()
    pre_list = []
    label_list = []

    with torch.no_grad():
        if config.network.multi_mode:
            text_inputs = classes.to(device)
            text_features = model_image.encode_text(text_inputs)
        for iii, (images, sam_images, class_id, text_feat) in enumerate(tqdm(val_loader)):
            
            if config.network.type == 'cnn_clip' or config.network.type == 'timesformer' or config.network.type == 'uniformer':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                sam_images = sam_images.view((-1,config.data.num_segments,3)+ sam_images.size()[-2:]) 
                images = rearrange(images, 'b t c h w -> b c t h w')
                sam_images = rearrange(sam_images, 'b t c h w -> b c t h w')
                images = images.cuda()
                sam_images = sam_images.cuda()
                text_feat = text_feat.cuda()                
                _, sam_feat = sam_model(sam_images)
                # image_embedding = model_image(images, sam_feat)
                cls_scores = model_image(images, sam_feat, text_feat)
            import ipdb; ipdb.set_trace()
            # [8, 17]
            class_id = [x.tolist() for x in class_id]
            class_id =torch.tensor(class_id).t()
            # [1. 17]
            cls_scores = torch.sigmoid(cls_scores)
            pre_list += cls_scores.tolist()
            label_list += class_id.tolist()
            
    mAP = MAP(label_list,pre_list,config.data.num_classes) * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: mAP: {}'.format(epoch, config.solver.epochs, mAP))
    return mAP

def validate_multicls_other(epoch, val_loader, classes, device, model_image, sam_model, config):
    model_image.eval()

    sam_model.eval()
    pre_list = []
    label_list = []

    with torch.no_grad():
        if config.network.multi_mode:
            text_inputs = classes.to(device)
            text_features = model_image.encode_text(text_inputs)
        for iii, (images, sam_images, class_id) in enumerate(tqdm(val_loader)):
            
            if config.network.type == 'cnn_clip' or config.network.type == 'timesformer' or config.network.type == 'uniformer':
                images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
                sam_images = sam_images.view((-1,config.data.num_segments,3)+ sam_images.size()[-2:]) 
                images = rearrange(images, 'b t c h w -> b c t h w')
                sam_images = rearrange(sam_images, 'b t c h w -> b c t h w')
                
                _, sam_feat = sam_model(sam_images)
                # image_embedding = model_image(images, sam_feat)
                cls_scores = model_image(images, sam_feat)
                
            class_id = [x.tolist() for x in class_id]
            class_id =torch.tensor(class_id).t()
            
            cls_scores = torch.sigmoid(cls_scores)
            pre_list += cls_scores.tolist()
            label_list += class_id.tolist()
            
    mAP = MAP(label_list,pre_list,config.data.num_classes) * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: mAP: {}'.format(epoch, config.solver.epochs, mAP))
    return mAP

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        # config = yaml.load(f)
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp_test', config['network']['type'], config['network']['arch'], config['data']['dataset'],args.log_time)
    # wandb.init(project=config['network']['type'],
              #  name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                        #  config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)
    clip_state_dict =  torch.load('/home/zx/Finegrained-Tubeproject-clip/ViT-B-16.pt', map_location="cpu")

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    if config.network.multi_mode:
      model_text = TextCLIP(model)
      model_text = model_text.cuda()
    
    model_image = ImageCLIP(model)
    model_image = model_image.cuda()
    fusion_model = fusion_model.cuda()
    # wandb.watch(model)
    # wandb.watch(fusion_model)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
      if config.network.multi_mode:
        model_text.float()
      model_image.float()
    else:
      if config.network.multi_mode:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
      clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes = text_prompt(val_data)
    if config.network.multi_mode:
      fg_model = finegrained_model(config.network.in_channels,config.data.num_classes,config.data.batch_size,config.network.if_softmax,config.network.fusion_type)
      fg_model = fg_model.cuda()

      text_inputs = classes.to(device)
      text_embedding = model_text(text_inputs)#torch.Size([cls_num, 512])
      # pdb.set_trace()
    else:
        fg_model = Single_Mode_Head(config.network.in_channels,config.data.num_classes,dropout_ratio=config.network.head_dropout)
        fg_model = fg_model.cuda()

    
    # prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)
    mAP = validate_multicls(start_epoch,val_loader, classes, device, model,fusion_model,fg_model, config)

if __name__ == '__main__':
    main()
