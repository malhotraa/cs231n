import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in np.arange(num_train):
    f = X[i].dot(W)
    f -= np.max(f)

    dW[:, y[i]] -= X[i]    
    for j in np.arange(num_classes):
      dW[:, j] += (np.exp(f[j]) / np.sum(np.exp(f))) * X[i]
    
    f_yi = f[y[i]]
    softmax = np.exp(f_yi)/ np.sum(np.exp(f))
    loss_i = - np.log (softmax)
    loss += loss_i
  
  dW += reg * 2 * W
  dW /= num_train
  loss += reg * np.sum(W * W)
  loss /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  probs = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
  probs_yis = probs[np.arange(scores.shape[0]), y]
  loss = - np.sum(np.log(probs_yis)) / num_train
  loss += reg * np.sum(W * W)
  
  # Subtract one from prob of correct classes as per gradient equation
  probs[range(scores.shape[0]), y] -= 1
  dW = X.T.dot(probs) / num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

