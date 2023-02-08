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


class SuperviseBCEMaskLoss(BaseLoss):
    def __init__(self, name='DEFAULT'):
        super().__init__(name)

    def forward(self, pred, target):
        val = torch.nn.functional.binary_cross_entropy(pred, target)
        return val


class NeighborGradientLoss(BaseLoss):
    def __init__(self, rad, name='NeighborGradientLoss', dist='l2'):
        super().__init__(name)
        assert rad > 0
        self.rad = rad
        self.crit = None
        if dist == 'l1':
            self.crit = torch.nn.L1Loss(reduction='none')
        elif dist == 'l2':
            self.crit = torch.nn.MSELoss(reduction='none')
        elif dist == 'smoothl1':
            self.crit = torch.nn.SmoothL1Loss(reduction='none')
        else:
            raise NotImplementedError(f'Unknown loss type: {dist}')

    def forward(self, depth, mask):
        """
            depth: [pch_len**2 * N, 1]
            mask: [pch_len**2 * N, 1]
        """
        pch_len = self.rad * 2 + 1
        depth_patch = depth.reshape(pch_len, pch_len, -1, 1)
        mask_patch = mask.reshape(pch_len, pch_len, -1, 1)

        x_grad = depth_patch[:-1, 1:, :, :] - depth_patch[:-1, :-1, :, :]
        x_error = self.crit(x_grad, torch.zeros_like(x_grad))
        x_mask = mask_patch[:-1, 1:, :, :] * mask_patch[:-1, :-1, :, :]

        y_grad = depth_patch[1:, :-1, :, :] - depth_patch[:-1, :-1, :, :]
        y_error = self.crit(y_grad, torch.zeros_like(y_grad))
        y_mask = mask_patch[1:, :-1, :, :] * mask_patch[:-1, :-1, :, :]

        grad_error = (x_error * x_mask + y_error * y_mask).sum()
        grad_base = (x_mask + y_mask).sum() + 1e-8

        val = grad_error / grad_base
        return val


class NeighborGradientLossWithEdge(NeighborGradientLoss):
    def __init__(self, rad, name='NeighborGradientLossWithEdge', dist='l2', sigma=0.4):
        super().__init__(rad, name, dist)
        self.dx_thd = 0.35
        self.dy_thd = 0.35
        self.sigma = sigma

    def forward(self, depth, mask, color=None):
        pch_len = self.rad * 2 + 1

        # Compute color gradient
        if color is not None:
            color_channel = color.shape[-1]
            color_patch = color.reshape(pch_len, pch_len, -1, color_channel)
            dx_color = torch.abs(color_patch[:-1, 1:, :, :] - color_patch[:-1, :-1, :, :]).sum(dim=-1, keepdim=True)
            dy_color = torch.abs(color_patch[1:, :-1, :, :] - color_patch[:-1, :-1, :, :]).sum(dim=-1, keepdim=True)
            dx_color[dx_color <= self.dx_thd] = 0.0
            dy_color[dy_color <= self.dy_thd] = 0.0
        else:
            dx_color = torch.Tensor([0.0]).to(depth.device)
            dy_color = torch.Tensor([0.0]).to(depth.device)

        depth_patch = depth.reshape(pch_len, pch_len, -1, 1)
        mask_patch = mask.reshape(pch_len, pch_len, -1, 1)
        x_grad = depth_patch[:-1, 1:, :, :] - depth_patch[:-1, :-1, :, :]
        x_error = self.crit(x_grad, torch.zeros_like(x_grad))
        x_mask = mask_patch[:-1, 1:, :, :] * mask_patch[:-1, :-1, :, :]
        x_mask *= torch.exp(- dx_color / (self.sigma ** 2))  # Color kernel

        y_grad = depth_patch[1:, :-1, :, :] - depth_patch[:-1, :-1, :, :]
        y_error = self.crit(y_grad, torch.zeros_like(y_grad))
        y_mask = mask_patch[1:, :-1, :, :] * mask_patch[:-1, :-1, :, :]
        y_mask *= torch.exp(- dy_color / (self.sigma ** 2))  # Color kernel

        grad_error = (x_error * x_mask + y_error * y_mask).sum()
        grad_base = (x_mask + y_mask).sum() + 1e-8

        val = grad_error / grad_base
        return val


class PeakEncourageLoss(BaseLoss):
    def __init__(self, name='PeakEncourageLoss'):
        super().__init__(name)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, weights):
        """
            weights: [N, samples]
        """
        max_idx = torch.argmax(weights, dim=1)
        res = self.cross_entropy(weights, max_idx)
        return res

