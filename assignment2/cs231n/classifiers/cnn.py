from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        
        # Input shape
        C, H, W = input_dim
        
        # Conv1 layer
        # Input size : (N,C,H,W)
        # Output size : (N,F,conv1_out_H,conv1_out_W)
        # Weights shape (F, C, HH, WW)
        stride = 1
        pad = (filter_size - 1) // 2
        F, C, HH, WW = num_filters, input_dim[0], filter_size, filter_size
        conv1_out_H = int(1 + (H + 2 * pad - HH) / stride)
        conv1_out_W = int(1 + (W + 2 * pad - WW) / stride)
        
        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params['b1'] = np.zeros((F), dtype=dtype)        
        
        # Pool layer
        # Input shape (N, F, conv1_out_H, conv1_out_W)
        # Output shape (N, F, pool_out_H, pool_out_W)
        # No params
        pool_H = 2
        pool_W = 2
        stride = 2
        pool_out_H = int(1 + (conv1_out_H - pool_H) / stride)
        pool_out_W = int(1 + (conv1_out_W - pool_W) / stride)
        
        # FC layer
        # Input shape (N, F * pool_out_H, pool_out_W)
        # Output shape (N, hidden_dim)
        # Weights shape ()
        self.params['W2'] = weight_scale * np.random.randn(F * pool_out_H * pool_out_W, hidden_dim)
        self.params['b2'] = np.zeros((hidden_dim), dtype=dtype)
        
        # Final layer
        # Input shape (N, hidden_dim)
        # Output shape (N, num_classes)
        # Weights shape (hidden_dim, num_classes)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros((num_classes), dtype=dtype)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_pool_out, W2, b2)
        final_layer_out, final_layer_cache = affine_forward(affine_relu_out, W3, b3)
        
        scores = final_layer_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        
        dx, dW3, db3 = affine_backward(dscores, final_layer_cache)
        dx, dW2, db2 = affine_relu_backward(dx, affine_relu_cache)
        dx, dW1, db1 = conv_relu_pool_backward(dx, conv_relu_pool_cache)
        
        loss += self.reg * np.sum(W1 * W1)
        loss += self.reg * np.sum(W2 * W2)
        loss += self.reg * np.sum(W3 * W3)
        
        grads['W3'] = dW3 + self.reg * 2 * W3
        grads['b3'] = db3
        grads['W2'] = dW2 + self.reg * 2 * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + self.reg * 2 * W1
        grads['b1'] = db1        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
