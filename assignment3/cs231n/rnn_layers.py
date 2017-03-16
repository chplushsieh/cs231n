import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  N, D = x.shape
  _, H = prev_h.shape

  # print 'N:', N
  # print 'D:', D
  # print 'H:', H

  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  h_score = prev_h.dot(Wh)
  x_score = x.dot(Wx)
  score = h_score + x_score + b
  next_h = np.tanh(score)

  cache = (x, prev_h, Wx, Wh, b, next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.

  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass

  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  x, prev_h, Wx, Wh, b, next_h = cache

  N, D = x.shape
  _, H = prev_h.shape

  # print 'N:', N
  # print 'D:', D
  # print 'H:', H

  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  dscore = dnext_h * (1 - next_h**2)  # (N, H)

  db = np.sum(dscore, axis=0)  # (H,)

  dx_score = dscore  # (N, H)

  # print 'dx_score:', dx_score.shape  # (N, H)
  # print 'x:', x.shape  # (N, D)
  dWx_per_datum = dx_score[:, np.newaxis, :] * x[:, :, np.newaxis]  # (N, D, H)
  dWx = np.sum(dWx_per_datum, axis=0)  # (D, H)

  dx_cross = dx_score[:, np.newaxis, :] * Wx[np.newaxis, :, :]  # (N, D, H)
  dx = np.sum(dx_cross, axis=2)  # (N, D)

  dh_score = dscore  # (N, H)

  # print 'dh_score:', dh_score.shape  # (N, H)
  # print 'prev_h:', prev_h.shape  # (N, H)
  dWh_per_datum = dh_score[:, np.newaxis, :] * prev_h[:, :, np.newaxis]  # (N, H, H)
  dWh = np.sum(dWh_per_datum, axis=0)  # (H, H)

  dprev_h_cross = dh_score[:, np.newaxis, :] * Wh[np.newaxis, :, :]  # (N, H, H)
  dprev_h = np.sum(dprev_h_cross, axis=2)  # (N, H)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.

  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  N, T, D = x.shape
  _, H = h0.shape

  # print 'N:', N
  # print 'T:', T
  # print 'D:', D
  # print 'H:', H

  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  h = np.zeros((N, T, H))
  step_cache = [None] * T

  ht = h0
  for t in range(T):
    # compute
    xt = x[:, t, :]
    h_next, cache_next = rnn_step_forward(xt, ht, Wx, Wh, b)

    # store output
    h[:, t, :]= h_next
    step_cache[t] = cache_next

    # update h for next round
    ht = h_next

  cache = D, step_cache
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)

  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  N, T, H = dh.shape

  D, step_cache = cache

  # print 'N:', N
  # print 'T:', T
  # print 'D:', D
  # print 'H:', H

  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  dx = np.zeros((N, T, D))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros((H,))

  dprev_h = np.zeros((N, H))  # for persisting dprev_h to the next round

  for t in reversed(range(T)):
    dht = dh[:, t, :] + dprev_h
    dxt, dprev_h, dWxt, dWht, dbt = rnn_step_backward(dht, step_cache[t])

    dx[:, t, :] = dxt
    dWx += dWxt
    dWh += dWht
    db  += dbt

  dh0 = dprev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  N, T = x.shape
  V, D = W.shape

  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################

  reshaped_x = x.reshape(N*T) # (N*T, )
  reshaped_out = W[reshaped_x] # (N*T, D)
  out = reshaped_out.reshape(N, T, D)

  cache = x, reshaped_x, W
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at

  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass

  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  x, reshaped_x, W = cache

  N, T, D = dout.shape
  V, _ = W.shape

  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  dW = np.zeros((V, D))

  reshaped_dout = dout.reshape(N*T, D)
  np.add.at(dW, reshaped_x, reshaped_dout)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  N, H = prev_h.shape

  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  h_score = prev_h.dot(Wh)
  x_score = x.dot(Wx)
  score = h_score + x_score + b  # (N, 4H)

  i = sigmoid(score[:,    :  H])  # (N, H)
  f = sigmoid(score[:,   H:2*H])
  o = sigmoid(score[:, 2*H:3*H])
  g = np.tanh(score[:, 3*H:4*H])

  next_c = f * prev_c + i * g
  next_h = o * np.tanh(next_c)

  cache = prev_h, prev_c, Wx, Wh, b, i, f, o, g, x, score, next_h, next_c
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.

  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  N, H = dnext_h.shape

  prev_h, prev_c, Wx, Wh, b, i, f, o, g, x, score, next_h, next_c = cache

  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################

  # dnext_h depends from dnext_c in forward pass
  dnext_c += o * (1 - np.tanh(next_c) ** 2) * dnext_h

  dprev_c = dnext_c * f

  di = dnext_c * g
  dscore1 = di * i*(1-i)

  df = dnext_c * prev_c
  dscore2 = df * f*(1-f)

  do = dnext_h * np.tanh(next_c)
  dscore3 = do * o*(1-o)

  dg = dnext_c * i
  dscore4 =  dg * (1 - g**2)

  dscore = np.concatenate((dscore1, dscore2, dscore3, dscore4), axis=1)  # (N, 4H)

  # the following is similar to rnn step

  db = np.sum(dscore, axis=0)  # (4H,)

  dx_score = dscore  # (N, 4H)

  dWx_per_datum = dx_score[:, np.newaxis, :] * x[:, :, np.newaxis]  # (N, D, 4H)
  dWx = np.sum(dWx_per_datum, axis=0)  # (D, 4H)

  dx_cross = dx_score[:, np.newaxis, :] * Wx[np.newaxis, :, :]  # (N, D, 4H)
  dx = np.sum(dx_cross, axis=2)  # (N, D)

  dh_score = dscore  # (N, 4H)

  dWh_per_datum = dh_score[:, np.newaxis, :] * prev_h[:, :, np.newaxis]  # (N, H, 4H)
  dWh = np.sum(dWh_per_datum, axis=0)  # (H, 4H)

  dprev_h_cross = dh_score[:, np.newaxis, :] * Wh[np.newaxis, :, :]  # (N, H, 4H)
  dprev_h = np.sum(dprev_h_cross, axis=2)  # (N, H)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.

  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.

  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)

  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]

  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)

  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape

  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)

  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]

  if verbose: print 'dx_flat: ', dx_flat.shape

  dx = dx_flat.reshape(N, T, V)

  return loss, dx
