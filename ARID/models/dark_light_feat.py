import sys

import torch
import torch.nn as nn


from .r2plus1d import r2plus1d_34_32_ig65m
from collections import OrderedDict

__all__ = ['dark_light_feat']





class dark_light_feat(nn.Module):
    def __init__(self, num_classes):
        super(dark_light_feat, self).__init__()
        self.hidden_size = 512

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])

        self.fc_action = nn.Linear(self.hidden_size, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.nobertpool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        # print(x.shape) #4, 3, 8, 112, 112
        x = self.features(x)  # x(b,512,8,7,7)
        x_feat = x
        # use cross attention here
        x = self.avgpool(x)  # b,512,8,1,1
        x = x.view(x.size(0), self.hidden_size, 8)  # x(b,512,8)
        x = x.transpose(1, 2)  # x(b,8,512)      
        x = x.mean(dim=1, keepdim=False)
        x = self.fc_action(x)  # b,11
        return x, x_feat

if __name__ == '__main__':

    model_path = '/data0/workplace/ActionCLIP-master/exp/clip_arid/R(2+1)D/ARID/20230703_234216/model_best.pth.tar'
    model = dark_light_feat(num_classes=11)
    checkpoint = torch.load(model_path)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])


    video = torch.randn(8, 3, 64, 112, 112)
    output, output_feat = model(video)
    print(output_feat.shape)



