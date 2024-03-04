#!/usr/bin/env python
import os
from collections import OrderedDict

import torch
from torch import nn, einsum
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.registry import register_model
from einops_exts import rearrange_many
from einops import rearrange, repeat

# cnn_clip with 
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
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)
        q = q * self.scale
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)


class KnowledgeAdapterBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, num_latents = 64, 
                 num_media_embeds = 4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(8, num_latents, d_model))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, d_model))
        self.norm_latents = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.cross_attn1 = PerceiverAttention(d_model)

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, visual_feat):
        visual_feat = rearrange(visual_feat, 'n (b t) d -> b t n d', t = 8)
        latents = repeat(self.latents, 't n d -> b t n d', b = visual_feat.shape[0])
        latents = self.norm_latents(latents)
        latents = self.cross_attn1(visual_feat, latents)
        latents = rearrange(latents, 'b t n d -> n (b t) d')
        latents = latents + self.drop_path(self.attention(self.ln_1(latents)))
        latents = latents + self.drop_path(self.mlp(self.ln_2(latents)))
        return latents


class TextAdapterBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, num_latents = 64, 
                 num_media_embeds = 4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, d_model))
        self.norm_latents = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.cross_attn1 = PerceiverAttention(d_model)

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, text_feat):       
        text_feat = rearrange(text_feat, 'b n d -> b 1 n d')
        latents = repeat(self.latents, 'n d -> b 1 n d', b = text_feat.shape[0])
        latents = self.norm_latents(latents)
        latents = self.cross_attn1(text_feat, latents)
        latents = rearrange(latents, 'b t n d -> n (b t) d')
        latents = latents + self.drop_path(self.attention(self.ln_1(latents)))
        latents = latents + self.drop_path(self.mlp(self.ln_2(latents)))
        return latents



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
        self.addition_block = KnowledgeAdapterBlock(d_model,n_head)
        self.cross_attn = PerceiverAttention(d_model)
        self.cross_attn_text = PerceiverAttention(d_model)
        self.text_block = TextAdapterBlock(d_model,n_head)

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x, clip_feat = None, text_feat = None, T = 8):
        
        if clip_feat == None and text_feat == None:
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape # 196 64 768(64 = 8 x 8) 已经把1拿出来了，怪不得卷积效果一般
            N = NT // T
            H = W = int(L ** 0.5)
            # 196, 64, 768 -> 8, 768, 8, 14, 14
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            # 8, 768, 8, 14, 14 -> 196, 64, 768
            tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
            
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            
            tmp_x = x[1:, :, :]
           
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
            
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        elif text_feat == None:
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape # 196 64 768(64 = 8 x 8) 已经把1拿出来了，怪不得卷积效果一般
            N = NT // T
            H = W = int(L ** 0.5)
            # 196, 64, 768 -> 8, 768, 8, 14, 14
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            # 8, 768, 8, 14, 14 -> 196, 64, 768
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
            clip_feat = self.addition_block(clip_feat)
            x, clip_feat = rearrange_many((x, clip_feat), 'n (b t) d -> b t n d', t = T)
            x = x + 0.1 * self.cross_attn(clip_feat, x)
            x = rearrange(x, 'b t n d ->  n (b t) d')
        
        elif clip_feat is not None and text_feat is not None:
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape # 196 64 768(64 = 8 x 8) 已经把1拿出来了，怪不得卷积效果一般
            N = NT // T
            H = W = int(L ** 0.5)
            # 196, 64, 768 -> 8, 768, 8, 14, 14
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            # 8, 768, 8, 14, 14 -> 196, 64, 768
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
            clip_feat = self.addition_block(clip_feat)
            text_feat = self.text_block(text_feat)
            x, clip_feat = rearrange_many((x, clip_feat), 'n (b t) d -> b t n d', t = T)
            text_feat = repeat(text_feat, 'n b d -> b 8 n d')
            # x = x + 0.1 * self.cross_attn(clip_feat, x)
            # 如何让cross_attn_text弄点新的东西
            x =  x + 0.05 * self.cross_attn(clip_feat, x) + 0.05 * self.cross_attn_text(text_feat, x)
            x = rearrange(x, 'b t n d ->  n (b t) d')
        else:
            # raise not implemented error
            raise NotImplementedError
        
        return x

# add feature here (caption and video feature)
class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0.,):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, drop_path=dpr[idx]))

    def forward(self, x, clip_feat, text_feat):
        for i in range(len(self.resblocks) - 3):
            x = self.resblocks[i](x)
        x = self.resblocks[-3](x, clip_feat[-3])
        x = self.resblocks[-2](x, clip_feat[-2])
        x = self.resblocks[-1](x, clip_feat[-1], text_feat)
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
        

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def forward(self, x, clip_feat, text_feat):
        
        x = self.conv1(x)  # shape = [*, width, grid, grid] 8, 768, 8, 14, 14

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)  # 64 196 768
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width] 64 197 768
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD 197 64 768
        x = self.transformer(x, clip_feat, text_feat)
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
    # width is dmodel
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
    # generate a tensor with the size of (8, 3, 8, 224, 224)
    video = torch.randn(32, 3, 8, 224, 224)
    text_feat = torch.zeros(32, 25, 768)
    visual_feat = torch.randn(197, 256, 768)
    last_feat = torch.randn(32, 1, 768)
    result = model(video, visual_feat, text_feat)
    # print(result.shape)
    # block = TextAdapterBlock(768, 8)
    
    # haha = block(text_feat)
    # print the max element in tensor
    # max_ele = torch.max(haha)
    print(result)
    
   
    
    
    

