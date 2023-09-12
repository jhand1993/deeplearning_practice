"""
This files provides tests for PyTorch models found in
'bic_pytorch' module.  Also tests the testing_loader module.
"""
import logging

import pytest
from torch import rand, manual_seed
from torch.optim import SGD
from torch.nn.functional import binary_cross_entropy_with_logits

from ..ptmodels import bic_pytorch as bp
from ..datasets import testing_loader

logger = logging.getLogger('__test_bic__')
logging.basicConfig(
    filename='logs/test_bic.log',
    encoding='utf-8',
    level=logging.DEBUG
)
logger.setLevel(logging.WARNING)


def test_forward_output_logistic() -> None:
    """ Tests the forward output value through the network.
        Implicitly tests the output shape as well.
    """
    manual_seed(0)
    x_in = rand((1, 3, 16, 16))
    x_out = bp.Logistic(n_in=768)(x_in)  # 16 * 16 * 3
    assert x_out.item() == pytest.approx(-0.1403, 1e-3)


def test_forward_output_perceptron() -> None:
    """ Tests the forward output value through the network.
        Implicitly tests the output shape as well.
    """
    manual_seed(0)
    x_in = rand((1, 3, 16, 16))
    n_layers = (768, 128, 16)
    x_out = bp.Perceptron(n_layers=n_layers)(x_in)  # 16 * 16 * 3
    assert x_out.item() == pytest.approx(-0.1588, 1e-3)


def test_forward_output_convnet() -> None:
    """ Tests the forward output value through the network.
        Implicitly tests the output shape as well.
    """
    manual_seed(0)
    # ConvNet is hard-coded to handle 128x128 images.
    x_in = rand((1, 3, 128, 128))
    x_out = bp.ConvNet(7, 3, 768, 16)(x_in)
    assert x_out.item() == pytest.approx(0.2774, 1e-3)


def test_TestDataSet() -> None:
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


def test_training() -> None:
    """ Tests training loop function bp.train_bic_model().
        Tests for all BIC models in bp module.
    """
    t_dl, v_dl = testing_loader.load_TestDataSets(
        8, 4, (3, 64, 64), (), 2, 2, scalar_label=True
    )
    n_in = 64 * 64 * 3
    test_l = bp.Logistic(n_in=n_in)
    test_p = bp.Perceptron(n_layers=(n_in, 1000, 100))
    test_c = bp.ConvNet(3, 3, 4 * 4 * 12, 16)
    opt_l = SGD(test_l.parameters(), lr=1e-3)
    opt_p = SGD(test_p.parameters(), lr=1e-3)
    opt_c = SGD(test_c.parameters(), lr=1e-3)
    loss = binary_cross_entropy_with_logits
    success_l = True
    success_p = True
    success_c = True

    # Logistics run.
    try:
        bp.train_bic_model(
            t_dl, v_dl, test_l, opt_l, loss, 1, 'cpu'
        )

    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)
        success_l = False

    # Perceptron run.
    try:
        bp.train_bic_model(
            t_dl, v_dl, test_p, opt_p, loss, 1, 'cpu'
        )

    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)
        success_p = False

    # ConvNet run.
    try:
        bp.train_bic_model(
            t_dl, v_dl, test_c, opt_c, loss, 1, 'cpu'
        )
        logger.warning('hello')

    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)
        success_c = False

    assert (success_l, success_p, success_c) == (1, 1, 1)
