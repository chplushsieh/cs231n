import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    dscores = np.zeros(num_classes)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dscores[j] = 1
        loss += margin

    yi = y[i]
    dscores[yi] = - np.sum(dscores)

    for j in xrange(num_classes):
      dW[:, j] += X[i] * dscores[j]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 0.5 * 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_features = X.shape[1]

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  syi = scores[np.arange(num_train), y]
  scores = (scores.T - syi).T  # transpose for broadcasting
  scores = scores + 1
  scores[scores < 0] = 0  # max(0, ) operation

  scores[np.arange(num_train), y] = 0  # for j==yi
  data_loss = np.sum(scores) / num_train

  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  dscores = np.zeros((num_train, num_classes))

  # To compute d(data_loss) / dscores
  dscores[scores > 0] = 1  # for j != yi
  dscores[np.arange(num_train), y] = - np.sum(dscores, axis=1)  # for j==yi

  # To compute ds / dW
  new_X = X[:, :, np.newaxis]  # new_X.shape:  (500, 3073, 1)
  new_dscores = dscores[:, np.newaxis, :]  # new_dscores.shape:  (500, 1, 10)
  ds = new_X * new_dscores  # dsi.shape:  (500, 3073, 10)

  ddata_loss = np.sum(ds, axis=0) / num_train
  dW += ddata_loss
  dW += 0.5 * 2 * reg * W  # d(reg_loss) / dW

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
