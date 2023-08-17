""" This files provides tests for PyTorch models found in 'ptmodel' module
"""
import pytest
from ptmodels import bic_pytorch as bp
from torch import rand, manual_seed


def test_forward_output():
    """ Tests the forward output value through the network. 
        Implicitly tests the output shape as well. 
    """
    manual_seed(0)
    x_in = rand((1, 3, 16, 16))
    x_out = bp.Logistic(n_in=768)(x_in) # 16 * 16 * 3
    assert x_out.item() == pytest.approx(-0.1403, 1e-3)


def test_forward_output():
    """ Tests the forward output value through the network. 
        Implicitly tests the output shape as well. 
    """
    manual_seed(0)
    x_in = rand((1, 3, 16, 16))
    n_layers = (768, 128, 16)
    x_out = bp.Perceptron(n_layers=n_layers)(x_in) # 16 * 16 * 3
    assert x_out.item() == pytest.approx(-0.1588, 1e-3)


def test_forward_output():
    """ Tests the forward output value through the network. 
        Implicitly tests the output shape as well. 
    """
    manual_seed(0)
    x_in = rand((1, 3, 128, 128)) # ConvNet is hard-coded to handle 128x128 images. 
    x_out = bp.ConvNet(7, 3, 16)(x_in)
    assert x_out.item() == pytest.approx(0.2774, 1e-3)


# if __name__ == '__main__':
#     LogistiscsTestCase().test_forward_output()
#     PerceptronTestCase().test_forward_output()
#     ConvNetTestCase().test_forward_output()
