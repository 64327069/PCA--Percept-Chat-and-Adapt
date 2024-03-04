# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch.nn as nn
from datasets.dataset_all_feat import Action_DATASETS_MOD
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt

from modules.finegrained_model import finegrained_model,Single_Mode_Head
from utils.KLLoss import KLLoss
from test_mod import validate, validate_multicls_all_feat
from utils.Augmentation import *
from utils.eval_mAP import MAP
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
import torch.nn.functional as F
import pdb
from clip.model_predict import predict_Transformer

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def get_feat(config):
    from clip.clip_extract import load_extract
    
    model, clip_state_dict = load_extract(config.network.arch,device='cuda',jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) 
    
    model_path = './63.4SR/model_best.pt'
    pretrain = torch.load(model_path)
    model.load_state_dict(pretrain['model_state_dict'], strict = False)
    model = torch.nn.DataParallel(model.visual).cuda()
    return model

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.full_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
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
    '''
    Parents,是否创建父目录,True等同于mkdir-p:False时,父目录不存在,则抛出fileNotfounderror。
    exist_ok参数,在3.5版本加入,flase时路径存在,抛出异常,True时候异常被忽略。
    '''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed_all(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    from clip.clip_mod import load_mod
    model, clip_state_dict = load_mod(config.network.arch,device=device,jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) 
    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header, 512 ,config.data.num_segments)
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    model_image = torch.nn.DataParallel(model.visual).cuda()
    sam_model = get_feat(config)
    
    train_data = Action_DATASETS_MOD(config.data.train_list,config.data.label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,cls_num=config.data.num_classes,random_shift=config.data.random_shift,
                       transform=transform_train)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    val_data = Action_DATASETS_MOD(config.data.val_list,config.data.label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,cls_num=config.data.num_classes,
                       transform=transform_val)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=False)
    model_image.float()
    sam_model.float()
    loss_multicls = nn.MultiLabelSoftMarginLoss(reduction='mean')
    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'], strict = False)
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'], strict = False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes = text_prompt_tube(train_data)
    fg_model = Single_Mode_Head(config.network.in_channels,config.data.num_classes,dropout_ratio=config.network.head_dropout)
    fg_model = fg_model.cuda()
    optimizer = _optimizer(config, model_image, fusion_model,fg_model)
    
    mAP = validate_multicls_all_feat(0, val_loader, classes, device, model_image,fusion_model,fg_model, sam_model, config)
    print('######map {} ######'.format(mAP))

if __name__ == '__main__':
    main()