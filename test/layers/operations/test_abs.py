import torch
import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerAbs(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerAbs, self).__init__()

    def forward(self, x):
        x = torch.abs(x)
        return x



@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_abs(change_ordering):
    model = LayerAbs()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
