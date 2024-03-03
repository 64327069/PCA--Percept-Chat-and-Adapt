# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu
import sys

import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def _optimizer(config, model, fusion_model, cd_model, prompt_learner, fine_grained_prompt):
    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()},
                                {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
                               lr=config.solver.lr, betas=(0.9, 0.98), eps=1e-8,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Adam')
    elif config.solver.optim == 'sgd':

        optimizer = optim.SGD([{'params': model.parameters()},
                               {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
                              config.solver.lr,
                              momentum=config.solver.momentum,
                              weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':

        vision_params = list(map(id, model.visual.parameters()))
        text_params = filter(lambda p: id(p) not in vision_params,
                             model.parameters())
        # block_uniformer_dec = list(map(id, model.visual.transformer.dec.parameters()))
        # block_uniformer_dpe = list(map(id, model.visual.transformer.dpe.parameters()))
        # vision_backbone = filter(lambda p: id(p) not in block_uniformer_dpe and id(p) not in block_uniformer_dec,
        #                      model.visual.parameters())
        vision_adapter_params = model.visual.named_parameters()
        vision_adapter_params = [item for name, item in vision_adapter_params if name.find('decoder') != -1]

        vision_adapter_params_id = list(map(id, vision_adapter_params))
        vision_backbone = filter(lambda p: id(p) not in vision_adapter_params_id,
                                 model.visual.parameters())


        optimizer = optim.AdamW([{'params': text_params},
                                 {'params': vision_backbone, 'lr': config.solver.lr * config.solver.ratio},
                                 {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
                                 {'params': cd_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
                                 {'params': prompt_learner.parameters(), 'lr': 0.0002},  # 8e-6 63   1e-7 62.8 0.005
                                 {'params': fine_grained_prompt.parameters(), 'lr': 0.0002},
                                 {'params': vision_adapter_params, 'lr': config.solver.lr * config.solver.f_ratio},
                                 ],
                                betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

        # optimizer = optim.AdamW([{'params': text_params},
        #                          {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio},
        #                          {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
        #                          {'params': cd_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
        #                          {'params': prompt_learner.parameters(), 'lr': 0.002},  # 8e-6 63   1e-7 62.8 0.005
        #                          {'params': fine_grained_prompt.parameters(), 'lr': 0.0002}
        #                          ],
        #                         betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
        #                         weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer


def _lr_scheduler(config, optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler
