import sys

import torch
from torch import nn
import torch.nn.functional as F


class CrossEn(nn.Module):
    def __init__(self, ):
        super(CrossEn, self).__init__()

    def forward(self, v2g_sim_matrix, l2g_sim_matrix):

        sim_loss = F.cross_entropy(v2g_sim_matrix, l2g_sim_matrix)
        return sim_loss


class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()

    def forward(self, v2g_sim_matrix, l2g_sim_matrix):
        logpt = torch.sigmoid(v2g_sim_matrix)
        # loss = F.binary_cross_entropy(logpt, l2g_sim_matrix, weight=weight)
        loss = F.binary_cross_entropy(logpt, l2g_sim_matrix)
        return loss


class Rankloss(nn.Module):
    def __init__(self):
        super(Rankloss, self).__init__()

    def forward(self, v2g_sim_matrix, l2g_sim_matrix):
        loss = nn.MarginRankingLoss(0.1)   ##  0.3 66.46634615384616
        tot_loss = 0.
        v2g_sim_matrix = F.softmax(v2g_sim_matrix, dim=-1)
        for idx, item in enumerate(v2g_sim_matrix):
            l = 0.
            for idy, sim in enumerate(item):
                l += loss(v2g_sim_matrix[idx][l2g_sim_matrix[idx]].unsqueeze(0), v2g_sim_matrix[idx][idy].unsqueeze(0),
                          torch.tensor(1, dtype=torch.float).to(v2g_sim_matrix.device).unsqueeze(0))
            tot_loss += l

        return tot_loss / v2g_sim_matrix.shape[0]



class MLT_loss(nn.Module):
    def __init__(self):
        super(MLT_loss, self).__init__()

    def forward(self, v2g_sim_matrix, l2g_sim_matrix):

        logpt = torch.sigmoid(v2g_sim_matrix)
        # loss = F.binary_cross_entropy(logpt, l2g_sim_matrix, weight=weight)

        sim_loss = F.cross_entropy(v2g_sim_matrix, l2g_sim_matrix)
        return sim_loss