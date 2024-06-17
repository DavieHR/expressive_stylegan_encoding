from easydict import EasyDict as edict
from .lpips import LPIPS
from .pixel_loss import L2Loss, L1Loss,MSELoss
from .frame_diff_based_loss import FDBLoss
from .face_parsing_loss import FaceParsingLoss
from .id_loss import IDLoss

class LossRegisterBase:
    def __init__(
                 self,
                 configure: edict
                ):

        assert isinstance(configure, edict), f"configure expected type is EasyDict, but {type(configure)}"
        assert hasattr(configure, "losses"), "losses attribute not exists in this configure."
        assert isinstance(configure.losses, list), "attribute 'losses' expected type is list."
    
        #TODO: pretty print configure.

        for _loss in configure.losses:
            loss_alias = _loss.alias
            _this_loss_config = dict()
            if hasattr(_loss, "config"):
                _this_loss_config = _loss.config
            
            loss_instantiate = eval(_loss.name)(**_this_loss_config)
            setattr(self, loss_alias, loss_instantiate)
            setattr(self, loss_alias + "_weight", _loss.weights)


    def forward(
                self,
                *args,
                **kwargs
               ):
        pass

    def __call__(
                 self,
                 *args,
                 is_gradient: bool = True,
                 **kwargs
                ):
        ret = self.forward(*args, **kwargs)
        assert isinstance(ret, dict), "expect return type is Dict."
        total = 0
        for k, v in ret.items():
            total += v

        if is_gradient:
            total.backward(retain_graph = True)
        ret["loss"] = total
        return ret

