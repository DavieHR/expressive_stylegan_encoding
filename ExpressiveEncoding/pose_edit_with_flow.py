import os
import sys
import torch
import numpy as np
from .StyleFlow.module.flow import cnf
where_am_i = os.path.dirname(os.path.realpath(__file__))

class PoseEdit:
    _zero_padding = torch.zeros(1, 18, 1)
    def __init__(self,
                 model_path = os.path.join(where_am_i, \
                              'third_party/models/modellarge10k.pt')
                ):
        self.cnf = cnf(512, '512-512-512-512-512', 17, 1)
        
        self.cnf.load_state_dict(torch.load(model_path))
        self.cnf.eval()
        self.cnf.to("cuda:0")
        for p in self.cnf.parameters():
            p.requires_grad = False
        self.reset()

    def reset(self):
        self.zflow = self._get_attribute_zflow()
        self.zflow.requires_grad = False   

    def _get_attribute_zflow(self):
        
        light_zflow = np.zeros((1,9,1,1), dtype = np.float32) 
        attribute_zflow = np.zeros((8,1), dtype = np.float32)

        attribute_zflow[0,0] = 0.0
        attribute_zflow[1,0] = 0.0
        attribute_zflow[2,0] = 0.0
        attribute_zflow[3,0] = 0.0
        attribute_zflow[4,0] = 0.0
        attribute_zflow[5,0] = 0.0
        attribute_zflow[6,0] = 55.0 # set value same as paper
        attribute_zflow[7,0] = 0.0

        zflow_array = np.concatenate([light_zflow, np.expand_dims(attribute_zflow, axis = (0, -1))], axis = 1)
        return torch.from_numpy(zflow_array).type(torch.FloatTensor).cuda()

    def _update_zflow(self,
                      yaw,
                      pitch
                      ):
        """Fix other attribute value,
           expose yaw and pitch.
           0-8: light
           9-16: attributes.
                 yaw and pitch index is 11, 12.
           
        """
        self.zflow[:, 11, 0, 0] = yaw
        self.zflow[:, 12, 0, 0] = pitch

    def __call__(
                self, 
                latent,
                yaw,
                pitch,
                is_w_space = False
                ):
        n = latent.shape[0]
        if self.zflow.shape[0] != n:
            self.reset()
            self.zflow = self.zflow.repeat(n,1,1,1)
        self._update_zflow(yaw, pitch)
        return self.cnf(latent, self.zflow, self._zero_padding.to('cuda'), is_w_space)[0]
