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
  num_class = W.shape[1]
  
  yhat = np.zeros(num_class)
  yt = np.zeros(num_class)
  z = np.zeros(num_class)
  
  
  for i in xrange(num_train):
      
      z[:] = X[i,:].dot(W) 
      den = 0.0
      yt[:] = 0.0
      yt[y[i]] = 1.0
      
      for j in xrange(num_class):
          yhat[j] = np.exp(z[j])
          den += yhat[j]
          
      yhat = yhat/den
      
      loss -= np.sum(yt*np.log(yhat))
   
      dW += np.dot(np.reshape(X[i,:],(-1,1)) , np.reshape((yhat - yt),(1,-1)) )
  
  loss = loss/num_train
  loss += reg*np.sum(W*W)
  
  dW = dW/num_train
  dW += 2*reg*W
       
  pass
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
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  z = np.dot(X,W)
  yt = np.zeros((num_train,num_class))
  yt[np.arange(num_train),y] = 1.0
  
  den = np.logaddexp.reduce(z,axis=1,dtype=np.float64)
  yhat = np.exp(z - den.reshape(-1,1))
  loss = -1*np.sum(yt*np.log(yhat))
  loss /= num_train
  loss += reg*np.sum(W*W)
  
  dW = (np.dot(X.T,(yhat - yt)))/num_train
  dW += 2.0*reg*W
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

