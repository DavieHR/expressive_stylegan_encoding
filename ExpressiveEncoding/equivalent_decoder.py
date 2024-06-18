import os
import sys
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "stylegan3"))
import copy
import torch
import numpy as np
from .decoder import StyleToRGBLayer, StyleSpaceSythesisLayer, StyleSpaceSythesisBlock, \
                     bias_act, misc, StyleSpaceDecoder

from .decoder import upfirdn2d as upf


from DeepLog import logger
#import upf._parse_padding, upf._get_filter_size, upf._parse_scaling

def upsample2d(x, conv_resample, up=2, padding=0, flip_filter=False, gain=1, impl='ref'):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = upf._parse_scaling(up)
    padx0, padx1, pady0, pady1 = upf._parse_padding(padding)

    #MOD: hard code 4x4.
    fw, fh = 4, 4#upf._get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, conv_resample, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy, impl=impl)
    #return upfirdn2d(x, conv_resample, up=up, padding=p, flip_filter=flip_filter, gain=4, impl=impl)

def upfirdn2d(x, conv, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='ref'):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    return _upfirdn2d_ref(x, conv, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn2d_ref(x, conv, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = upf._parse_scaling(up)
    downx, downy = upf._parse_scaling(down)
    padx0, padx1, pady0, pady1 = upf._parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    #assert upW >= f.shape[-1] and upH >= f.shape[0]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    x = conv(x)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    conv,                       # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_conv   = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    """
    # disable these code feature.
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]
    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I
    """
    out_channels, in_channels, kh, kw = conv.weight.shape
    if up > 1:
        w = conv.weight.transpose(0,1) * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    else:
        w = conv.weight * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]

    """
    w = w * (1 / w.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
    styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I
    """
    # Calculate per-sample weights and demodulation coefficients.
    dcoefs = torch.ones(batch_size, out_channels, 1, 1).to(x)
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        dcoefs = dcoefs.reshape(batch_size, -1, 1, 1)

    """
    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x
    """
    
    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)

    #misc.assert_shape(x, [batch_size, in_channels, None, None])
    #x = x.reshape(1, -1, *x.shape[2:])
    #w = w.reshape(-1, in_channels, kh, kw)
    #x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    #x = x.reshape(batch_size, -1, *x.shape[2:])
    # 3 1 3 1
    # 1 0 1 0
    # 0 0
    if demodulate:
        x = conv(x * styles.reshape(batch_size, -1, 1, 1)) * dcoefs
    else:
        x = conv(x * styles.reshape(batch_size, -1, 1, 1))
    if up > 1:
        x = upfirdn2d(x = x, conv = resample_conv, padding=[1,1,1,1], gain=4, flip_filter=False)
    if noise is not None:
        x = x.add_(noise)
    return x

class _adaptor:
    def __init__(self, this):
        if hasattr(this, "weight"):
            out_channels, in_channels, k, k = this.weight.shape
            if not hasattr(this, "up"):
                up = 1
            else:
                up = this.up

            if not hasattr(this, "down"):
                down = 1
            else:
                down = this.down
            weight = this.weight
            if up == 1 and down == 1:
                pad = k // 2
                self.conv = torch.nn.Conv2d(in_channels, out_channels, k, 1, pad, bias = False)
            elif up > 1:
                weight = this.weight.transpose(0,1)
                self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, k, 2, padding = (0,0), bias = False)
            self.conv.weight.data = weight.data
        else:
            out_channels = 3
    
        if hasattr(this, "resample_filter"):
            _filter = this.resample_filter * 4
            # _filter = this.resample_filter
            _filter = _filter[np.newaxis, np.newaxis].repeat([out_channels, 1] + [1] * _filter.ndim)
            self.conv_resample = torch.nn.Conv2d(out_channels, out_channels, 4, 1, 0, bias = False, groups = out_channels)
            self.conv_resample.weight.data = _filter.data# * 4

class EquivalentStyleToRGBLayer(StyleToRGBLayer, _adaptor):
    def __init__(self, module):
        StyleToRGBLayer.__init__(self, module = module)
        _adaptor.__init__(self, self)
        self.register_buffer("weight_gain_buffer", torch.tensor(self.weight_gain, dtype = torch.float32))

    def forward(self, x, w, fused_modconv=True, encoded_styles=None):
        styles = w
        tmp_s = styles * self.weight_gain_buffer
        x = modulated_conv2d(x=x, conv = self.conv, styles=tmp_s, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

class EquivalentStyleSpaceSythesisLayer(StyleSpaceSythesisLayer, _adaptor):
    def __init__(self, module):
        StyleSpaceSythesisLayer.__init__(self, module)
        _adaptor.__init__(self, self)
        self.register_buffer("act_gain_buffer", torch.tensor(self.act_gain, dtype = torch.float32))
        if self.conv_clamp is not None:
            self.register_buffer("conv_clamp_buffer", torch.tensor(self.conv_clamp, dtype = torch.float32))
        

    def forward(self, x, w, noise_mode='const', fused_modconv=True, gain=1, encoded_styles=None):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        # misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution]) # not need to be squre 
        styles = w
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, conv = self.conv,styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_conv=self.conv_resample, flip_weight=flip_weight, fused_modconv=fused_modconv)

        #act_gain = self.act_gain * gain
        #act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        act_gain = self.act_gain_buffer
        act_clamp = self.conv_clamp_buffer if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

class EquivalentStyleSpaceSythesisBlock(StyleSpaceSythesisBlock, _adaptor):
    def __init__(self,
                 module
                ):
        StyleSpaceSythesisBlock.__init__(self, module)
        _adaptor.__init__(self, self)

        if hasattr(self, "conv0"):
            self.conv0 = EquivalentStyleSpaceSythesisLayer(self.conv0)
        self.conv1 = EquivalentStyleSpaceSythesisLayer(self.conv1)
        if hasattr(self, "torgb"):
            self.torgb = EquivalentStyleToRGBLayer(self.torgb)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=True, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        w_iter = iter(ws)
        force_fp32 = True
        if ws[0].device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws[0].shape[0], 1, 1, 1])
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
            img = upsample2d(img, self.conv_resample)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            #img = y
            img = img.add_(y) if img is not None else y
            
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

class EquivalentStyleSpaceDecoder(StyleSpaceDecoder):
    def __init__(
                 self, stylegan_path: str = None,
                 synthesis: object = None,
                 to_resolution: int = 1024
                ):
        super().__init__(stylegan_path, synthesis, to_resolution)
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            block_forked = EquivalentStyleSpaceSythesisBlock(block)
            setattr(self, f"b{res}", block_forked)

    def load_state_dict(self, state_dict, strict = True):

        from collections import OrderedDict

        _new_ordered_dict = OrderedDict()

        current_ordered_dict = self.state_dict()
        for k, v in state_dict.items():
            new_key = k
            if 'weight' in k and 'affine' not in k:
                new_key = k.replace('weight', 'conv.weight')
            if 'resample_filter' in k:
                new_key = k.replace('resample_filter', 'conv_resample.weight')
            if new_key in current_ordered_dict:
                current_v = current_ordered_dict[new_key]
                _old_shape = v.shape
                _new_shape = current_v.shape

                if 'resample_filter' in k:
                    _new_ordered_dict[new_key] = v[np.newaxis, np.newaxis].repeat(_new_shape[0], _new_shape[1], 1, 1) * 4
                elif "conv0" in k and 'weight' in k:
                    _new_ordered_dict[new_key] = v.transpose(0,1)
                else:
                    _new_ordered_dict[new_key] = v
                _new_ordered_dict[k] = v
        return super().load_state_dict(_new_ordered_dict, strict)

