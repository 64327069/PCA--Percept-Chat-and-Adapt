import sys

import torch
import torch.nn as nn

from .r2plus1d import r2plus1d_34_32_ig65m
from collections import OrderedDict

__all__ = ['dark_light']





class dark_light(nn.Module):
    def __init__(self, num_classes, length, both_flow):
        super(dark_light, self).__init__()
        self.hidden_size = 512

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
        
    def forward(self, x):
        _, x = x
        # print(x.shape) #4, 3, 8, 112, 112
        x = self.features(x)  # x(b,512,8,7,7)
        # use cross attention here
        x = self.avgpool(x)  # b,512,8,1,1
        x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
        x = x.transpose(1, 2)  # x(b,8,512)      
        x = x.mean(dim=1, keepdim=False)
        x = self.fc_action(x)  # b,11
        return x

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    model = dark_light(11, 64, True)
    video = torch.randn(1, 3, 64, 112, 112)
    flops = FlopCountAnalysis(model, video)
    print(flop_count_table(flops, max_depth=1))



