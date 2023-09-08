""" Tests variational autoencoders found in ptmodels -> vae_pytorch.py.
"""
import logging

# import pytest
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import SGD

from ..ptmodels import vae_pytorch as vp
from ..datasets import testing_loader as tl

logger = logging.getLogger('__test_vae__')
logger.setLevel(logging.DEBUG)

# Instantiate testing DataLoaders.
test_shape = (1, 28, 28)
test_label_shape = (3,)
t_dl, v_dl = tl.load_TestDataSets(
    8, 8, test_shape, test_label_shape, 4, 4
)
# Grab first input batch as a testing batch.
for xb, _ in t_dl:
    x = xb

# Test logg function alias.
test_loss = binary_cross_entropy_with_logits


def test_DenseAE() -> None:
    """ Test DenseAE functionality.
    """

    # Instantiate model.
    test_model = vp.DenseAE(
        [784, 16], test_label_shape[0], (28, 28), 1
    )
    test_opt = SGD(test_model.parameters(), lr=1e-3)
    x_lat = test_model.get_encoding(x)
    x_pred = test_model.get_decoding(x_lat)

    # make sure encoders and decoders are working.
    assert x_lat.shape == (4, 3)
    assert x_pred.shape == (4, *test_shape)  # append batch size to tuple

    success = True
    try:
        vp.train_AE(
            t_dl, v_dl, test_model, test_opt, test_loss,
            1, 'cpu'
        )

    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)
        success = False

    assert success
