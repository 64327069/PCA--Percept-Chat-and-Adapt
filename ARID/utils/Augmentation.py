# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu
import sys

import numpy as np
import torch
from datasets.transforms_ss import *
from torchvision.transforms import RandAugment
import utils.augmix_ops as augmentations
from torchvision.transforms import transforms


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


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


def randAugment(transform_train, config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train


############################################post_pretraining transform####################################
# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
        GroupRandomCrop(224),
        GroupRandomHorizontalFlip(False),
    ])


def augmix(v, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(v)
    x_processed = preprocess(x_orig)

    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    # mix = torch.zeros_like(x_processed)

    # for i in range(3):
    #     x_aug = x_orig.copy()
    #     for _ in range(np.random.randint(1, 4)):
    #         x_aug = GroupTransform(RandAugment(1, 9))(x_aug)
    #     mix += w[i] * preprocess(x_aug)
    # mix = m * x_processed + (1 - m) * mix
    mix = GroupTransform(RandAugment(2, 9))(x_orig)
    mix = preprocess(mix)

    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False,
                 severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, v):
        video = self.preprocess(self.base_transform(v))

        views = [augmix(v, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        # print(len(views))  ## 63
        # print(views[0].shape)  ## torch.Size([72, 224, 224])
        # print(video.shape)  ### torch.Size([72, 224, 224])

        return [video] + views
