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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
      score = X[i].dot(W)
      score-= np.argmax(score)
      exp_score = np.exp(score)
      sf_sum = np.sum(exp_score)
      loss  -= np.log(exp_score[y[i]]/sf_sum)
      #compute dW loops:
      for j in range(num_classes):
        if j != y[i]:
          dW[:,j] += exp_score[j] / sf_sum * X[i]
        else:
          dW[:,j] += -X[i]+exp_score[j]/sf_sum* X[i]
          
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += 2*reg * W
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
  scores=X.dot(W)
  #broadcast need reshape for the list
  scores1=scores-np.max(scores,axis =1).reshape(num_train,1)
  coefficient=np.exp(scores1)
  sf_sum = np.sum(coefficient,axis=1)
  #LOSS
  loss += -np.sum(np.log(coefficient[np.arange(num_train), y]/sf_sum))
  #cannot use reshape(-1,1)!!!  or it will become(500,500)  by broadcasting
  loss =  -np.sum(np.log(coefficient[np.arange(num_train),y]/sf_sum))
  #dW
  reg_coefficient=coefficient/sf_sum.reshape(-1,1)
  reg_coefficient[np.arange(num_train), y] -= 1
  dW = X.T.dot(reg_coefficient)
  # regularization
  loss/=num_train
  dW/=num_train
  loss +=reg*np.sum(W*W)
  dW +=2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

