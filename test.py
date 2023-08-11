""" This files provides tests for PyTorch models found in 'ptmodel' module
"""
from unittest import TestCase
from ptmodels import bic_pytorch as bp
from torch import rand, manual_seed


class LogistiscsTestCase(TestCase):
    """ 
    Test to make sure logistics classifer network outputs the correct shape. 
    Note that bp.Logistsics flattens an input image before a forward call. 
    """

    def test_forward_output(self):
        """ Tests the forward output value through the network. 
            Implicitly tests the output shape as well. 
        """
        manual_seed(0)
        x_in = rand((1, 3, 16, 16))
        x_out = bp.Logistic(n_in=768)(x_in) # 16 * 16 * 3
        self.assertAlmostEqual(x_out.item(), -0.1403, places=3)


class PerceptronTestCase(TestCase):
    """ Test to make sure the three layer perceptron binary classifer outputs
        the correct shape and values. 
    """
    def test_forward_output(self):
        """ Tests the forward output value through the network. 
            Implicitly tests the output shape as well. 
        """
        manual_seed(0)
        x_in = rand((1, 3, 16, 16))
        n_layers = (768, 128, 16)
        x_out = bp.Perceptron(n_layers=n_layers)(x_in) # 16 * 16 * 3
        self.assertAlmostEqual(x_out.item(), -0.1588, places=3)


class ConvNetTestCase(TestCase):
    """ Test to make sure ConvNet binary classifer outputs the correct shape and 
        values. 
    """
    def test_forward_output(self):
        """ Tests the forward output value through the network. 
            Implicitly tests the output shape as well. 
        """
        manual_seed(0)
        x_in = rand((1, 3, 128, 128)) # ConvNet is hard-coded to handle 128x128 images. 
        x_out = bp.ConvNet(7, 3, 16)(x_in)
        self.assertAlmostEqual(x_out.item(), 0.2774, places=3)


if __name__ == '__main__':
    LogistiscsTestCase().test_forward_output()
    PerceptronTestCase().test_forward_output()
    ConvNetTestCase().test_forward_output()
