import numpy as np
from random import shuffle

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
  dim = X.shape[1]
  num_class = W.shape[1]

  score = X.dot(W)
  for i in xrange(num_train):
    es = [ np.exp(score[i, j]) for j in xrange(num_class) ]
    sigma_es = np.sum(es)
    pyi = es[y[i]] / sigma_es
    loss += - np.log(pyi) # natural log

    # compute gradient
    # ref: http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    ds = [ es[j] / sigma_es for j in xrange(num_class) ]
    ds[y[i]] -= 1

    dW += np.outer(X[i], ds)

  dW /= num_train
  loss /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += 0.5 * 2 * reg * W

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_class = W.shape[1]

  score = X.dot(W)

  es = np.exp(score)
  sigma_es = np.sum(es, axis=1)
  p = es / (sigma_es[:, np.newaxis])

  # compute loss
  esy = es[np.arange(num_train), y]
  py = esy / sigma_es
  L = - np.log(py)
  loss = np.sum(L) / num_train

  # compute dL / dscores
  ds = p
  ds[np.arange(num_train), y] -= 1

  # compute dL / dW
  X = X[:, :, np.newaxis]
  ds = ds[:, np.newaxis, :]
  dLdW = X * ds

  dW += np.sum(dLdW, axis=0)
  dW /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += 0.5 * 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
