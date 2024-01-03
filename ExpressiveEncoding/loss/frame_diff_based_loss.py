import torch
from typing import Tuple, List

from .lpips import LPIPS
from .pixel_loss import L2Loss

class LPIPSDiffLoss(torch.nn.Module):
    def __init__(
                 self,
                 lpips_instance: LPIPS = None
                ):
        super().__init__()

        if lpips_instance is not None:
            self.net = lpips_instance.net
            self.lin = lpips_instance.lin


    def forward(self, 
                x1: torch.Tensor, x2: torch.Tensor,
                y1: torch.Tensor, y2: torch.Tensor,
                lpips_instance : LPIPS = None
               ) -> torch.Tensor:

        if lpips_instance is not None:
            self.net = lpips_instance.net
            self.lin = lpips_instance.lin

        feat_x1, feat_x2 = self.net(x1), self.net(x2)
        feat_y1, feat_y2 = self.net(y1), self.net(y2)
        diff_x = [(fx - fy) for fx, fy in zip(feat_x1, feat_x2)]
        diff_y = [(fx - fy) for fx, fy in zip(feat_y1, feat_y2)]
        diff = [(fx - fy) ** 2 for fx, fy in zip(diff_x, diff_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        return torch.sum(torch.cat(res, 1), dim = 1).reshape(-1)


class FDBLoss(torch.nn.Module):
    def __init__(self,
                 lpips = None,
                ):
        super().__init__()
        self.l2_loss = L2Loss()
        self.lpips_diff_loss = LPIPSDiffLoss(lpips)

    def forward(self, 
                x1: torch.Tensor, x2: torch.Tensor,
                y1: torch.Tensor, y2: torch.Tensor,
                lpips_instance : LPIPS = None
               ):

        mse_loss = self.l2_loss(x1 - x2, y1 - y2)
        lpips_loss = self.lpips_diff_loss(x1, x2, y1, y2, lpips_instance = lpips_instance)
        return mse_loss + lpips_loss




