import sys

import torch
import torch.nn as nn

# from r2plus1d import r2plus1d_34_32_ig65m
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from timm.models.registry import register_model
from einops_exts import rearrange_many
from einops import rearrange, repeat
import torch.hub


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
    

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.addition_block = KnowledgeAdapterBlock(512,8)
        self.cross_attn = PerceiverAttention(512)
        self.cross_attn_text = PerceiverAttention(512)
        self.text_block = TextAdapterBlock(512, 8)
        
        

    def forward(self, x, visual_feat = None, text_feat = None):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if visual_feat is not None and text_feat is None:
            B, C, T, H, W = out.shape
            out, visual_feat = rearrange_many((out, visual_feat), 'b c t h w -> b (t h w) c')
            visual_feat = self.addition_block(visual_feat)
            out =  out + 0.05 * self.cross_attn(visual_feat, out)
            out = rearrange(out, 'b (t h w) c -> b c t h w', t = T, h  = H, w = W)
        elif visual_feat is not None and text_feat is not None:
            
            B, C, T, H, W = out.shape
            out, visual_feat = rearrange_many((out, visual_feat), 'b c t h w -> b (t h w) c')
            visual_feat = self.addition_block(visual_feat)
            text_feat = self.text_block(text_feat)
            out =  out + 0.05 * self.cross_attn_text(text_feat, out) + 0.05 * self.cross_attn(visual_feat, out)
            out = rearrange(out, 'b (t h w) c -> b c t h w', t = T, h  = H, w = W)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.
        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x, visual_feat = None, text_feat = None):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x, visual_feat)
        x = self.layer4[1](x, visual_feat)
        x = self.layer4[2](x, visual_feat, text_feat)
        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def r2plus1d_34_32_ig65m(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M model for clips of length 32 frames.
    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 359, "pretrained on 359 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_32_ig65m",
                       pretrained=pretrained, progress=progress)

def r2plus1d_34(num_classes, pretrained=False, progress=False, arch=None):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    if pretrained:
        model_urls="https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth"
        state_dict = torch.hub.load_state_dict_from_url(model_urls,
                                                        progress=progress)
        model.load_state_dict(state_dict, strict = False)

    return model

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
        
        # nn.init.constant_(self.to_q.weight, 0)
        # nn.init.constant_(self.to_kv.weight, 0)
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
        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        q = q * self.scale
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)
        
    
class KnowledgeAdapterBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, num_latents = 128, 
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
    
    def forward(self, visual_feat):
        latents = repeat(self.latents, 'n d -> b n d', b = visual_feat.shape[0])
        latents = self.norm_latents(latents)
        latents = self.cross_attn1(visual_feat, latents)
        latents = latents + self.drop_path(self.attention(self.ln_1(latents)))
        latents = latents + self.drop_path(self.mlp(self.ln_2(latents)))
        return latents


class TextAdapterBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, num_latents = 96, 
                 num_media_embeds = 4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, d_model))
        self.norm_latents = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
        self.features = r2plus1d_34_32_ig65m(359, pretrained=True, progress=True)
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
        for param in self.features.parameters():
            param.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        
        
    def forward(self, x, visual_feat = None, text_feat = None):
        x = self.features(x, visual_feat, text_feat)  # x(b,512,8,7,7)
        x = self.avgpool(x)  # b,512,8,1,1
        x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
        x = x.transpose(1, 2)  # x(b,8,512)      
        x = x.mean(dim=1, keepdim=False)
        x = self.fc_action(x)  # b,11
        return x
    
if __name__ == '__main__':
    model = dark_light(11)
    video = torch.randn(8, 3, 64, 112, 112)
    video_feat = torch.randn(3, 8, 512, 8, 7, 7)
    text_feat = torch.randn(8, 20, 768)
    output = model(video, video_feat)
    print(output.shape)
