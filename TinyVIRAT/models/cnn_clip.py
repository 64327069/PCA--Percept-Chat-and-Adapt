#!/usr/bin/env python
import os
from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.registry import register_model


MODEL_PATH = '/mnt/lustre/share_data/likunchang/model/clip_visual_encoder'
_MODELS = {
    "ViT-B/16": os.path.join(MODEL_PATH, "vit_b16.pth"),
    "ViT-L/14": os.path.join(MODEL_PATH, "vit_l14.pth"),
    "ViT-L/14_336": os.path.join(MODEL_PATH, "vit_l14_336.pth"),
}


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class Local_MHRA(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__() 

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), stride=(1, 1, 1), padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.lmhra1 = Local_MHRA(d_model)
        self.lmhra2 = Local_MHRA(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    # TODO: if frames is not 8, please change here
    def forward(self, x, T = 8):
        # in this place can use checkpoint to reduce the memory usage
        
        tmp_x = x[1:, :, :]
        L, NT, C = tmp_x.shape
        N = NT // T
        H = W = int(L ** 0.5)
        tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
        tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
        x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        
        tmp_x = x[1:, :, :]
        L, NT, C = tmp_x.shape
        N = NT // T
        H = W = int(L ** 0.5)
        tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
        tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
        x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0.,):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, drop_path=dpr[idx]))

    def forward(self, x):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim, 
        kernel_size=1, num_frames=8, fc_drop_rate=0., drop_path=0, num_classes=400,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3, width, 
            (kernel_size, patch_size, patch_size), 
            (kernel_size, patch_size, patch_size), 
            (0, 0, 0),  bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        
        self.transformer = Transformer(width, layers, heads, drop_path=drop_path)

        self.proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(fc_drop_rate),
            nn.Linear(width, num_classes)
        )
        
    def get_parameters(self, mode):
        assert mode in ('proj', 'lmhra1', 'lmhra2', 'backbone')
        if mode == 'proj':
            return self.proj.parameters()
        elif mode == 'lmhra1':
            # return the parameters of ResidualAttentionBlock lmhra1
            para_list = [self.transformer.resblocks[i].lmhra1.parameters() for i in range(len(self.transformer.resblocks))]
            return para_list
        elif mode == 'lmhra2':
            # return the parameters of ResidualAttentionBlock lmhra1
            para_list = [self.transformer.resblocks[i].lmhra2.parameters() for i in range(len(self.transformer.resblocks))]
            return para_list
        else:
            raise ValueError('mode must be backbone or classifer')

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        x = self.transformer(x)


        x = x[0].view(B, T, -1).mean(1)
        
        x = self.proj(x)

        return x



def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, input_resolution=224, patch_size=16, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    pos_embed_checkpoint = state_dict['positional_embedding']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = (input_resolution // patch_size) ** 2
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        print(f'Pos_emb from {orig_size} to {new_size}')
        extra_tokens = pos_embed_checkpoint[:1]
        pos_tokens = pos_embed_checkpoint[1:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        state_dict['positional_embedding'] = new_pos_embed
    
    model.load_state_dict(state_dict, strict=False)


@register_model
def cnn_clip_mean_b16(
    pretrained_path, input_resolution=224, kernel_size=1,
    center=True, num_frames=8, fc_drop_rate=0., drop_path=0., num_classes=400
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=16, 
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames,
        fc_drop_rate=fc_drop_rate, drop_path=drop_path,
        num_classes=num_classes
    )
    if pretrained_path is not None:
        print('load pretrained weights')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model


@register_model
def cnn_clip_mean_l14(
    pretrained=True, input_resolution=224, kernel_size=1,
    center=True, num_frames=8, fc_drop_rate=0., drop_path=0., num_classes=400
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        fc_drop_rate=fc_drop_rate, drop_path=drop_path,
        num_classes=num_classes
    )
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14"], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


@register_model
def clip_mean_l14_336(
    pretrained=True, input_resolution=336, kernel_size=1,
    center=True, num_frames=8, fc_drop_rate=0., drop_path=0., num_classes=400
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14, 
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        fc_drop_rate=fc_drop_rate, drop_path=drop_path,
        num_classes=num_classes
    )
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14_336"], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()

if __name__ == '__main__':
    model = cnn_clip_mean_b16(pretrained_path = '/opt/data/private/workplace/ResKD-main/checkpoints/clip_vit_mean_b16_k400.pth')
    model.get_parameters('proj')

