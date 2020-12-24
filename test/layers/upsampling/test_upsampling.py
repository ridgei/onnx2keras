import torch.nn as nn
import numpy as np
from torch.nn.functional import bilinear
import pytest

from test.utils import convert_and_test


class LayerUpsample(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self, size, scale_factor, mode):
        super(LayerUpsample, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.upsample(x)
        return x


class FInterpolate(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self, mode, size=None, scale_factor=None):
        super(FInterpolate, self).__init__()
        self.mode = mode
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        from torch.nn import functional as F
        return F.interpolate(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
@pytest.mark.parametrize('size,scale_factor', [(None, 2), ((128, 128), None)])
def test_f_interpole(change_ordering, mode, size, scale_factor):
    model = FInterpolate(mode=mode, size=size, scale_factor=scale_factor)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 64, 64))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
@pytest.mark.parametrize('size,scale_factor', [(None, 2), ((128, 128), None)])
def test_layer_upsamle(change_ordering, mode, size, scale_factor):
    model = LayerUpsample(mode=mode, size=size, scale_factor=scale_factor)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 64, 64))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)

