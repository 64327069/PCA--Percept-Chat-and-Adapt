import sys

import torch
import torch.nn as nn

from .r2plus1d import r2plus1d_34_32_ig65m
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from timm.models.registry import register_model
from einops_exts import rearrange_many
from einops import rearrange, repeat

__all__ = ['dark_light']


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = True)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = True)
        self.to_out = nn.Linear(inner_dim, dim, bias = True)
        
        nn.init.constant_(self.to_q.weight, 0)
        nn.init.constant_(self.to_kv.weight, 0)
        # nn.init.constant_(self.to_out.weight, 0)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        # extract batch size, sequence length, and number of heads
        b, m, h = *x.shape[:2], self.heads
         # apply linear transformation to the latents to obtain the query tensor
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        # subtract the maximum similarity value across the key sequence to prevent 
        # numerical instability and apply softmax to obtain attention weights
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        # 注意这个attention的softmax，这就是导致没法收敛的原因
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        # 聚合multi head
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)
        
        
# learnable query 从外面输进去
class KnowledgeAdapterBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, num_latents = 64, 
                 num_media_embeds = 4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
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
        # self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.cross_attn1 = PerceiverAttention(d_model)
   
    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    # TODO: if frames is not 8, please change here
    def forward(self, visual_feat):
        latents = repeat(self.latents, 'n d -> b n d', b = visual_feat.shape[0])
        latents = self.norm_latents(latents)
        latents = self.cross_attn1(visual_feat, latents)
        # latents = latents.squeeze(1)
        latents = latents + self.drop_path(self.attention(self.ln_1(latents)))
        latents = latents + self.drop_path(self.mlp(self.ln_2(latents)))
        return latents


class TextAdapterBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, num_latents = 32, 
                 num_media_embeds = 4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, d_model))
        self.norm_latents = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head,bias = False)
        self.linear = nn.Linear(768, d_model,bias = False)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4, bias = False)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model, bias = False))
        ]))
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        self.attn_mask = attn_mask
        self.cross_attn1 = PerceiverAttention(d_model)

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    # TODO: if frames is not 8, please change here
    def forward(self, text_feat):
               
        latents = repeat(self.latents, 'n d -> b n d', b = text_feat.shape[0])
        latents = self.norm_latents(latents)
        
        text_feat = self.linear(text_feat)
        latents = self.cross_attn1(text_feat, latents)
        # latents = latents.squeeze(1)
        
        latents = latents + self.drop_path(self.attention(self.ln_1(latents)))
        latents = latents + self.drop_path(self.mlp(self.ln_2(latents)))        
        return latents

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)





class dark_light(nn.Module):
    def __init__(self, num_classes):
        super(dark_light, self).__init__()
        self.hidden_size = 512
        self.length = 64
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        # 预训练
        self.features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])

        self.fc_action = nn.Linear(self.hidden_size, num_classes)
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        self.addition_block = KnowledgeAdapterBlock(512,8)
        self.cross_attn = PerceiverAttention(512)
        self.cross_attn_text = PerceiverAttention(512)
        self.text_block = TextAdapterBlock(512, 8)
        
    def forward(self, x, visual_feat = None, text_feat = None):

        x = self.features(x)  # x(b,512,8,7,7)
        # use cross attention here
        if visual_feat is not None and text_feat is None:
            B, C, T, H, W = x.shape
            x, visual_feat = rearrange_many((x, visual_feat), 'b c t h w -> b (t h w) c')
            # 32 196 512
            visual_feat = self.addition_block(visual_feat)
            x =  x + 0.05 * self.cross_attn(visual_feat, x)
            x = rearrange(x, 'b (t h w) c -> b c t h w', t = T, h  = H, w = W)
        
        if visual_feat is not None and text_feat is not None:
            B, C, T, H, W = x.shape
            text_feat = self.text_block(text_feat)
            x, visual_feat = rearrange_many((x, visual_feat), 'b c t h w -> b (t h w) c')
            # 32 196 512
            visual_feat = self.addition_block(visual_feat)
            x =  x + 0.05 * self.cross_attn(visual_feat, x) + 0.05* self.cross_attn_text(text_feat, x)
            x = rearrange(x, 'b (t h w) c -> b c t h w', t = T, h  = H, w = W)
        
        if visual_feat is None and text_feat is not None:
            B, C, T, H, W = x.shape
            text_feat = self.text_block(text_feat)
            x = rearrange(x, 'b c t h w -> b (t h w) c')
            # 32 196 512
            x =  x +  0.01 * self.cross_attn_text(text_feat, x)
            x = rearrange(x, 'b (t h w) c -> b c t h w', t = T, h  = H, w = W)
            
        x = self.avgpool(x)  # b,512,8,1,1
        x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
        x = x.transpose(1, 2)  # x(b,8,512)      
        x = x.mean(dim=1, keepdim=False)
        x = self.fc_action(x)  # b,11
        return x
    
if __name__ == '__main__':
    model = dark_light(11)
    video = torch.randn(8, 3, 64, 112, 112)
    video_feat = torch.randn(8, 512, 8, 7, 7)
    text_feat = torch.randn(8, 20, 768)
    output = model(video, video_feat, text_feat)
    print(output.shape)
    # prin



