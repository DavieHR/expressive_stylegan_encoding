from torch.nn import MSELoss

class L2Loss(MSELoss):
    def __init__(self):
        super().__init__(reduction = 'none')

    def forward(self, x, y):
        return super().forward(x, y).mean(dim = (1,2,3))
