# -*- coding: utf-8 -*-

# - Package Imports - #
import torch


# - Coding Part - #
class BaseLoss(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class NaiveLoss(BaseLoss):
    def __init__(self, name='NAIVE'):
        super().__init__(name)

    def forward(self, loss):
        return loss


class SuperviseDistLoss(BaseLoss):
    """L1 or L2 loss for supervision"""
    def __init__(self, name='DEFAULT', dist='l1'):
        super().__init__(name)
        self.dist = dist
        self.crit = None
        if dist == 'l1':
            self.crit = torch.nn.L1Loss(reduction='none')
        elif dist == 'l2':
            self.crit = torch.nn.MSELoss(reduction='none')
        elif dist == 'smoothl1':
            self.crit = torch.nn.SmoothL1Loss(reduction='none')
        elif dist == 'bce':
            self.crit = torch.nn.BCELoss(reduction='none')
        else:
            raise NotImplementedError(f'Unknown loss type: {dist}')

    def forward(self, pred, target, mask=None):
        """
        disp_prd: [N, 1, H, W]
        """
        err_map = self.crit(pred, target)
        if mask is None:
            mask = torch.ones_like(pred)
        val = (err_map * mask).sum() / (mask.sum() + 1e-8)
        return val, err_map
