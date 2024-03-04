import torch
from torch import nn

from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet

from models.clip import clip_mean_b16
from models.mod_clip3 import cnn_clip_mean_b16
#from models.new_csn import ir_csn_152
from opts import parse_opts
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

def BatchFormer(x, y, encoder, is_training):
    if not is_training:
        return x, y
    pre_x = x
    x = encoder(x.unsqueeze(1)).squeeze(1)
    x = torch.cat([pre_x, x], dim = 0)
    y = torch.cat([y, y], dim = 0)
    return x, y

def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True
        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'i3d', 'video_swin','resnet2p1d101','uniformer', 'ircsn152','ipcsn152', 'clip', 'cnn_clip', 'dcnn_clip', 'timesformer'
    ]
    
    if opt.model == 'clip':
        if not opt.sub_path or opt.resume_path :
            model = clip_mean_b16(pretrained_path = None, num_classes = opt.n_classes)
        else:
            model = clip_mean_b16(pretrained_path = opt.pretrain_path, num_classes = opt.n_classes)
        
        model.proj = nn.Sequential(
            nn.LayerNorm(model.proj[2].in_features),
            nn.Dropout(get_dropout_rate()),
            nn.Linear(model.proj[2].in_features, 26)
        )
        
    
    elif opt.model == 'cnn_clip':
        if not opt.sub_path or opt.resume_path :
            model = cnn_clip_mean_b16(input_resolution = opt.sample_size,pretrained_path = None, num_classes = opt.n_classes)
        else:
            model = cnn_clip_mean_b16(input_resolution = opt.sample_size,pretrained_path = opt.pretrain_path, num_classes = opt.n_classes)
        
        model.proj = nn.Sequential(
            nn.LayerNorm(model.proj[2].in_features),
            nn.Dropout(get_dropout_rate()),
            nn.Linear(model.proj[2].in_features, 26)
        )    
    return model

def get_dropout_rate():
    opt = parse_opts()
    return opt.drop_out_rate

def get_batchformer():
    opt = parse_opts()
    return opt.batchformer

def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        if model_name == 'i3d':
            model.replace_logits(157)
            model.load_state_dict(torch.load(pretrain_path))
            model.replace_logits(n_finetune_classes)
        else:
            print('loading pretrained model {}'.format(pretrain_path))
            pretrain = torch.load(pretrain_path, map_location='cpu')
            
            tmp_model = model
            if model_name == 'uniformer':
                tmp_model.head = nn.Linear(tmp_model.head.in_features, n_finetune_classes)

            else:

                tmp_model.fc = nn.Sequential(nn.Dropout(get_dropout_rate()), 
                                             nn.Linear(tmp_model.fc.in_features, n_finetune_classes))

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
