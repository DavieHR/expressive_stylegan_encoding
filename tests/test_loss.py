import os
import sys

sys.path.insert(0, os.getcwd())
import yaml
import torch
import pytest 
from ExpressiveEncoding.loss import LossRegisterBase, edict


@pytest.mark.xy
def test_loss():
    class LossRegister(LossRegisterBase):
        def forward(self, x, y):
            l2_loss = self.l2(x, y)
            lpips_loss = self.lpips(x,y)
    
            return {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss
                   }
    with open("./tests/loss.yaml", "r") as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    loss_register = LossRegister(config)


    x = torch.randn(1,3,512,512).to("cuda:0")
    y = torch.randn(1,3,512,512).to("cuda:0")
    ret = loss_register(x, y, is_gradient = False)
    print(ret)

@pytest.mark.xy2
def test_loss():
    class LossRegister(LossRegisterBase):
        def forward(self, x1, x2, y1, y2):
            l2_loss = self.l2(x2, y2)
            lpips_loss = self.lpips(x2,y2)
            fdb_loss = self.fdb(x1, x2, y1, y2, lpips_instance = self.lpips)
    
            return {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss,
                     "fdb_loss": fdb_loss
                   }
    with open("./tests/loss.yaml", "r") as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    loss_register = LossRegister(config)


    x1 = torch.randn(1,3,512,512).to("cuda:0")
    x2 = torch.randn(1,3,512,512).to("cuda:0")
    y1 = torch.randn(1,3,512,512).to("cuda:0")
    y2 = torch.randn(1,3,512,512).to("cuda:0")
    ret = loss_register(x1,x2,y1,y2, is_gradient = False)
    print(ret)
