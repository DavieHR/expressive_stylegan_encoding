from torch.nn import MSELoss,L1Loss

class L1Loss(L1Loss):
    def __init__(self):
        super().__init__(reduction = 'none')

    def forward(self, x, y):
        n = x.dim()
        return super().forward(x, y).mean(dim = tuple(list(range(1,n))))

class L2Loss(MSELoss):
    def __init__(self):
        super().__init__(reduction = 'none')

    def forward(self, x, y):
        return super().forward(x, y).mean(dim = (1,2,3))
