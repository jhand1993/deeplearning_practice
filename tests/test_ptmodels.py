""" This files provides tests for PyTorch models found in 'ptmodel' module
"""
from unittest import TestCase
from ptmodels import bic_pytorch as bp
from torch import Size, rand, manual_seed

test_seed = manual_seed(0)

class LogistiscsTestCase(TestCase):
    """ 
    Test to make sure logistics classifer network outputs the correct shape. 
    bp.Logistsics flattens an input image before a forward push. 
    """

    def test_forward_output(self):
        """ Tests the forward output value through the network. 
            Implicitly tests the output shape as well. 
        """
        x_in = rand((10, 10, 3))
        x_out = bp.Logistic(n_in=300)(x_in) # 10 * 10 * 3
        self.assertEqual(x_out.item, 0.4610461890697479)
