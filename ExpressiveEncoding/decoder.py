import os
import sys
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "stylegan3"))
import torch
import dnnlib
import legacy
import copy

from DeepLog import logger

from training.networks_stylegan2 import SynthesisNetwork, SynthesisLayer, SynthesisBlock, \
                                        ToRGBLayer, misc, modulated_conv2d, \
                                        bias_act, upfirdn2d 

def load_model(network_pkl, device = "cpu"):
    if network_pkl.endswith("pkl"): 
        with dnnlib.util.open_url(network_pkl) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    elif network_pkl.endswith("pt"):
        G = torch.load(network_pkl)
    return G

class CopyLayer(torch.nn.Module):
    def __init__(
                 self, 
                 module_copy
                ):
        for _attr in module_copy.__dir__():
            if not hasattr(self, _attr):
                value = getattr(module_copy, _attr)
                setattr(self, _attr, value)

class StyleToRGBLayer(CopyLayer):
    def __init__(self, module):
        super().__init__(module)
        output_channels = self.weight.shape[1]
        """
        self.affine_delta = torch.nn.Linear(output_channels, output_channels)
        torch.nn.init.eye_(self.affine_delta.weight)
        torch.nn.init.zeros_(self.affine_delta.bias)
        """

    def forward(self, x, w, fused_modconv=True, encoded_styles=None, **kwargs):
        if isinstance(w, tuple):
            w, delta = w
            #delta_affine = self.affine_delta(delta)
            #w = self.affine_delta(w + delta)
            w = w + delta#self.affine(w) + delta
        styles = w

        tmp_s = styles * self.weight_gain  
        x = modulated_conv2d(x=x, weight=self.weight, styles=tmp_s, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

class StyleSpaceSythesisLayer(CopyLayer):
    def __init__(self, module):
        super().__init__(module)
        output_channels = self.affine.bias.shape[0]
        # self.affine_only_bias = torch.nn.Parameter(torch.zeros(output_channels, dtype = torch.float32))       
        """
        self.affine_delta = torch.nn.Linear(output_channels, output_channels)
        torch.nn.init.eye_(self.affine_delta.weight)
        torch.nn.init.zeros_(self.affine_delta.bias)
        """

    def forward(self, x, w, noise_mode='const', fused_modconv=True, gain=1, encoded_styles=None, **kwargs):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        # misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution]) # not need to be squre 
        # styles = w + self.affine_only_bias.to(x)
        if isinstance(w, tuple):
            w, delta = w
            w = w + delta#self.affine(w) + delta
            #delta_affine = self.affine_delta(delta)
            #w = self.affine_delta(w + delta)

        styles = w 
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

class StyleSpaceSythesisBlock(CopyLayer):
    def __init__(self,
                 module
                ):
        super().__init__(module)

        if hasattr(self, "conv0"):
            self.conv0 = StyleSpaceSythesisLayer(self.conv0)
        self.conv1 = StyleSpaceSythesisLayer(self.conv1)
        if hasattr(self, "torgb"):
            self.torgb = StyleToRGBLayer(self.torgb)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=True, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        w_iter = iter(ws) 

        force_fp32 = True
        #if w_iter[0].device.type != 'cuda':
        #    force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)
        n = ws[0].shape[0] if isinstance(ws[0], torch.Tensor) else ws[0][0].shape[0]


        # Input.
        if self.in_channels == 0:
            if x is None:
                x = self.const.to(dtype=dtype, memory_format=memory_format)
                x = x.unsqueeze(0).repeat([n, 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

class StyleSpaceDecoder(CopyLayer):
    def __init__(
                 self,
                 stylegan_path: str = None,
                 synthesis: object = None,
                 to_resolution: int = 1024,
                 device='cuda:0',
                ):
        if synthesis is not None:
            module = synthesis
        else:
            _base_model = copy.deepcopy(load_model(stylegan_path))
            module = _base_model.synthesis
        super().__init__(module)

        if synthesis is None:
            self.mapping = _base_model.mapping
        self.block_resolutions = [x for x in self.block_resolutions if x <= to_resolution]
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            block_forked = StyleSpaceSythesisBlock(block)
            setattr(self, f"b{res}", block_forked)

    def _group_latent(
                      self,
                      ws
                     ):
        ws_group = []
        group_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            group_num = block.num_conv + block.num_torgb
            ws_group.append(ws[group_idx:group_idx + group_num])
            group_idx += group_num
        return ws_group

    def forward(self, ws,  **block_kwargs):
        if isinstance(ws, torch.Tensor):
            ws = self.get_style_space(ws)

        insert_feature = block_kwargs.get("insert_feature", None)

        if not isinstance(ws[0], list):
            ws = self._group_latent(ws)
        x = img = None
        for res, cur_ws in zip(self.block_resolutions, ws):
            if insert_feature is not None and f'{res}' in insert_feature.keys():
                x = insert_feature[f"{res}"]
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def get_conv_weight(self, i, l):
        i_sum = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res==4: 
                if i_sum == i:
                    return block.conv1.weight[:, l, ...]
                i_sum += 1
                if i_sum == i:
                    return block.torgb.weight[:, l, ...]
                i_sum += 1
            else:

                if i_sum == i:
                    return block.conv0.weight[:, l, ...]
                i_sum += 1
                if i_sum == i:
                    return block.conv1.weight[:, l, ...]
                i_sum += 1
                if i_sum == i:
                    return block.torgb.weight[:, l, ...]
                i_sum += 1
        raise RuntimeError(f"{i} not in list.")

    def get_style_space(self, ws):
        i=0
        styles_space = []
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            styles = []
            if res==4: 
                s=block.conv1.affine(ws[:,i])
                b,o = s.shape
                styles += [s.reshape(b,o)]
                i+=1
                s=block.torgb.affine(ws[:,i])
                b,o = s.shape
                styles += [s.reshape(b,o)]
            else:
                s=block.conv0.affine(ws[:,i])
                b,o = s.shape
                styles += [s.reshape(b,o)]
                i+=1
                s=block.conv1.affine(ws[:,i])
                b,o = s.shape
                styles += [s.reshape(b,o)]
                i+=1
                s=block.torgb.affine(ws[:,i])
                b,o = s.shape
                styles += [s.reshape(b,o)]
            styles_space += styles
        return styles_space

    def get_base_code(self):
        return self.b4.const
