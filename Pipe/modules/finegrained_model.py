import torch
from torch import nn
from mmcv.cnn import normal_init
import pdb




class finegrained_model(nn.Module):
    def __init__(self, in_channels, num_classes, batch_size, if_softmax, fusion_type):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.if_softmax = if_softmax
        self.fusion_type = fusion_type
        if self.fusion_type == 'concat':
          self.fc = nn.Linear(self.num_classes, self.batch_size)
          self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
          self.fc.weight.data.normal_(mean=0.0, std=0.02)
          self.fc_cls.weight.data.normal_(mean=0.0, std=0.02)
        elif self.fusion_type == 'hadama':
          self.fc_cls = nn.Linear(self.in_channels, 1)
          self.fc_cls.weight.data.normal_(mean=0.0, std=0.02)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_img, x_txt):
        # pdb.set_trace()
        if self.fusion_type == 'concat':
          x_txt = x_txt.permute(1,0)
          x_txt = self.fc(x_txt.float())
          x_txt = x_txt.permute(1,0)
          x = torch.cat([x_img,x_txt],dim=1)#torch.Size([32, 1024])
          cls_score = self.fc_cls(x.float())
        elif self.fusion_type == 'hadama':
          x_txt = x_txt.unsqueeze(0)
          x_txt = x_txt.expand(self.batch_size,self.num_classes,self.in_channels)
          x_img = x_img.unsqueeze(1)
          x_img = x_img.expand(self.batch_size,self.num_classes,self.in_channels)
          x = x_img * x_txt
          cls_score = self.fc_cls(x.float())
          cls_score = cls_score.squeeze(-1)
        elif self.fusion_type == 'matmul':
          cls_score = torch.mm(x_img,x_txt.t())

        if self.if_softmax:
          cls_score = self.softmax(cls_score)
        
        return cls_score

class Single_Mode_Head(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_ratio=0.8,
                 init_std=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)


    def forward(self, x_img):
        # pdb.set_trace()
        # ([N,in_channels])
        if self.dropout is not None:
            x = self.dropout(x_img.float())
        else:
            x = x_img.float()
        cls_score = self.fc_cls(x)
        
        return cls_score