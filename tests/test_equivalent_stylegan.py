import os
import sys
import torch
import pytest
sys.path.insert(0, os.getcwd())


from ExpressiveEncoding.equivalent_decoder import EquivalentStyleToRGBLayer, EquivalentStyleSpaceDecoder, \
                                                  EquivalentStyleSpaceSythesisBlock, EquivalentStyleSpaceSythesisLayer

from ExpressiveEncoding.decoder import StyleToRGBLayer, StyleSpaceDecoder, \
                                       StyleSpaceSythesisBlock, StyleSpaceSythesisLayer, load_model
from ExpressiveEncoding.train import stylegan_path

eq_decoder = EquivalentStyleSpaceDecoder(stylegan_path = stylegan_path).to("cuda")
decoder = StyleSpaceDecoder(stylegan_path = stylegan_path)

@pytest.mark.decoder
def test_equivalent_decoder():
    _input = torch.randn(1,18,512).to("cuda")
    ss = eq_decoder.get_style_space(_input)
    _output_eq = eq_decoder(ss)
    _output = decoder(ss)

    diff = torch.abs(_output - _output_eq)

    print(f"diff max {diff.max()} min {diff.min()} avg {diff.mean()}")

@pytest.mark.load
def test_load():
    snapshots_path = "./results/pivot_007/snapshots/22.pth"
    state_dict = torch.load(snapshots_path)
    decoder.load_state_dict(state_dict)
    eq_decoder.load_state_dict(state_dict, False)

    _input = torch.randn(1,18,512).to("cuda")
    ss = eq_decoder.get_style_space(_input)
    _output_eq = eq_decoder(ss)
    _output = decoder(ss)

    diff = torch.abs(_output - _output_eq)

    print(f"diff max {diff.max()} min {diff.min()} avg {diff.mean()}")
    

