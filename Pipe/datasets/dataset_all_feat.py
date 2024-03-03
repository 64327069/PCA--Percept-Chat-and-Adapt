# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import json
import math
import torch
# from RandAugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row, cls_num, max_ele = None, pred = None):
        self._data = row
        self.cls_num = cls_num
        self.max_ele = max_ele
        self.pred = pred

    @property
    def path(self):
        return self._data[0]
    
    @property
    def sam_path(self): 
        sam_path = '/opt/data/private/Data/ECCV2022/red_sam'
        pic_path = self._data[0].split('/')[-1]
        return sam_path + '/' + pic_path

    @property
    def num_frames(self):
        # this way has changed
        return int(self._data[1])
        # return int(47)

    @property
    def label(self):
        def index2onehot(index,cls_num):
            onehot = [0]*cls_num
            for i in index:
                onehot[i] = 1
            return onehot
        return index2onehot([int(x) for x in self._data[2:]],self.cls_num)

    @property
    def get_npy_path(self):
        npy_dir = '/opt/data/private/workplace/pipe2/pip_feat'
        if self.pred is None:
            npy_path = '_'.join([str(x) for x in self._data[2:]]) + '.npy'
            
        else:
            if self.max_ele < -0.5:
                npy_path = self.path.split('/')[-1] + '.npy'
                
            else :
                npy_path = '_'.join([str(x) for x in self.pred]) + '.npy'
                
        fin_path = os.path.join(npy_dir, npy_path)
        if os.path.exists(fin_path):
            return fin_path

        else:
            print(fin_path)
            return os.path.join(npy_dir, 'empty.npy')
       
                



class Action_DATASETS_MOD(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', cls_num = 17, transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.cls_num = cls_num
        self.train_or_val = 'train' if 'train' in list_file else 'val'
        
        if self.index_bias is None:
            if self.image_tmpl == "{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        if 'train' in self.list_file:
            self.video_list = [VideoRecord(x.strip().split(' '),self.cls_num) for x in open(self.list_file)]
        else:
            all_anno = json.load(open(self.list_file))
            self.video_list = [VideoRecord([anno['path'],anno['num_frames']] + anno['target'], 
                                           self.cls_num, 
                                           anno['output'], 
                                           anno['pred']) for anno in all_anno]
            

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(num_frames // 2),
                    num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(num_frames),
                randint(num_frames,
                        size=self.total_length - num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, num_frames):
        if self.num_segments == 1:
            return np.array([num_frames //2], dtype=np.int64) + self.index_bias
        
        if num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), num_frames) + self.index_bias
            return np.array([i * num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int64) + self.index_bias
        offset = (num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int64) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices_ori = self._sample_indices(record.num_frames) if self.random_shift else self._get_val_indices(record.num_frames)
        segment_indices_sam = self._sample_indices(47) if self.random_shift else self._get_val_indices(47)
        
        return self.get(record, segment_indices_ori, segment_indices_sam)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record, indices_ori, indices_sam):
        images = list()
        sam_images = list()
        for i, seg_ind in enumerate(indices_ori):
            p = int(seg_ind)
            p1 = int(indices_sam[i])
            try:
                
                seg_imgs = self._load_image(record.path, p)
                sam_imgs = self._load_image(record.sam_path, p1)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices_ori))
                raise
            images.extend(seg_imgs)
            sam_images.extend(sam_imgs)
        # process here
        process_data = self.transform(images)
        process_data_sam = self.transform(sam_images)
        return process_data, process_data_sam, record.label, torch.tensor(np.load(record.get_npy_path))

    def __len__(self):
        return len(self.video_list)
    
if __name__ == '__main__':
    from dotmap import DotMap
    from torch.utils.data import DataLoader
    import yaml
    import tqdm
    import time
    config = '/opt/data/private/workplace/pipe2/exp_tube/tube_train_1.yaml'
    with open(config, 'r') as f:
        config = yaml.full_load(f)
    config = DotMap(config)
    from Augmentation_copy import *
    def get_augmentation(training, config):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = config.data.input_size * 256 // 224
        if training:

            unique = torchvision.transforms.Compose([GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
                                                    GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
                                                    GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                            saturation=0.2, hue=0.1),
                                                    GroupRandomGrayscale(p=0.2),
                                                    GroupGaussianBlur(p=0.0),
                                                    GroupSolarization(p=0.0)]
                                                    )
        else:
            unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                    GroupCenterCrop(config.data.input_size)])

        common = torchvision.transforms.Compose([Stack(roll=False),
                                                ToTorchFormatTensor(div=True),
                                                GroupNormalize(input_mean,
                                                                input_std)])
        return torchvision.transforms.Compose([unique, common])

    def randAugment(transform_train,config):
        print('Using RandAugment!')
        transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
        return transform_train
    
    
    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)
    
    train_data = Action_DATASETS_MOD(config.data.train_list,config.data.label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,cls_num=config.data.num_classes,random_shift=config.data.random_shift,
                       transform=transform_train)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=16,shuffle=True,pin_memory=False,drop_last=True)
    val_data = Action_DATASETS_MOD(config.data.val_list,config.data.label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,cls_num=config.data.num_classes,
                       transform=transform_val)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=16,shuffle=False,pin_memory=False,drop_last=True)
    print('all_done')
    begin = time.time()
    
    for kkk,(images,sam_images,list_id, text_feat) in enumerate(val_loader):
        images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
        print(images.shape, sam_images.shape, type(list_id), text_feat.shape)
        print('{} {}'.format(kkk, len(train_loader)))
    end = time.time()
    print('using {}'.format(end - begin))

