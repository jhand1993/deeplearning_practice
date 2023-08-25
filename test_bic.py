"""
This files provides tests for PyTorch models found in
'bic_pytorch' module.  Also tests the testing_loader module.
"""
import pytest
from ptmodels import bic_pytorch as bp
from datasets import testing_loader
from torch import rand, manual_seed


def test_forward_output_logistic():
    """ Tests the forward output value through the network.
        Implicitly tests the output shape as well.
    """
    manual_seed(0)
    x_in = rand((1, 3, 16, 16))
    x_out = bp.Logistic(n_in=768)(x_in)  # 16 * 16 * 3
    assert x_out.item() == pytest.approx(-0.1403, 1e-3)


def test_forward_output_perceptron():
    """ Tests the forward output value through the network.
        Implicitly tests the output shape as well.
    """
    manual_seed(0)
    x_in = rand((1, 3, 16, 16))
    n_layers = (768, 128, 16)
    x_out = bp.Perceptron(n_layers=n_layers)(x_in)  # 16 * 16 * 3
    assert x_out.item() == pytest.approx(-0.1588, 1e-3)


def test_forward_output_convnet():
    """ Tests the forward output value through the network.
        Implicitly tests the output shape as well.
    """
    manual_seed(0)
    # ConvNet is hard-coded to handle 128x128 images.
    x_in = rand((1, 3, 128, 128))
    x_out = bp.ConvNet(7, 3, 16)(x_in)
    assert x_out.item() == pytest.approx(0.2774, 1e-3)


def test_TestDataSet():
    """ Tests TestDataSet shape outputs.
    """
    size = 4
    feature_shape = (4, 4)
    label_shape = (2,)
    ds = testing_loader.TestDataSet(
        size, feature_shape, label_shape
    )
    test_x, test_y = ds.__getitem__(0)
    assert test_x.shape == feature_shape
    assert test_y.shape == label_shape

# def test_training():
#     """ Tests training loop function bp.train_bic_model().
#         Tests for all BIC models in bp module.
#     """
#     t_dl, v_dl = testing_loader.load_TestDataSets(
#         8, 4, (3, 4, 4), (2,), 2, 2
#     )
#     success = True
#     try:



if __name__ == '__main__':
    pass
