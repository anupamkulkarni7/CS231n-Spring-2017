from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        C = W2.shape[1]
        reg = self.reg
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        
        z1 = np.dot(X.reshape(N,-1),W1) + b1
        a1 = np.maximum(z1,0.0)
    
        z2 = a1.dot(W2) + b2
        den = np.logaddexp.reduce(z2,axis=1)
        a2 = np.exp(z2 - den.reshape(-1,1))
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return z2

        loss, grads = 0, {}
        
        yoh = np.zeros((N,C))
        yoh[np.arange(N),y] = 1
        
        loss = -1*np.sum(np.log(a2)*yoh)/N
        loss += 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
        
        #loss2,crap = softmax_loss(z2,y)
        #print("ACB",loss2)
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
    
        gradz2 = a2 - yoh  #N x C
    
        gradb2 = np.sum(gradz2,axis=0)/N #C x 1
        gradW2 = np.dot(a1.T,gradz2)/N    #H x C
    
        grada1 = np.dot(gradz2,W2.T)
        z1t = (z1>0.0).astype(np.float64)
        gradz1 = grada1*z1t
    
        gradb1 = np.sum(gradz1, axis=0)/N
        gradW1 = np.dot((X.reshape(N,-1)).T,gradz1)/N
    
        grads['W1'] = gradW1 + reg*W1
        grads['b1'] = gradb1
        grads['W2'] = gradW2 + reg*W2
        grads['b2'] = gradb2
  
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dims[0]) #*np.sqrt(2.0/input_dim)
        self.params['b1'] = np.zeros(hidden_dims[0])
        
        for hd in np.arange(1,len(hidden_dims)):
            nhw = 'W'+ str(hd+1)
            nhb = 'b'+ str(hd+1)
            self.params[nhw] = weight_scale * np.random.randn(hidden_dims[hd-1], hidden_dims[hd])
            self.params[nhb] = np.zeros(hidden_dims[hd])
        
        hd = len(hidden_dims) #output layer
        nhw = 'W'+ str(hd+1)
        nhb = 'b'+ str(hd+1)
        self.params[nhw] = weight_scale * np.random.randn(hidden_dims[hd-1],num_classes)
        self.params[nhb] = np.zeros(num_classes)
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        loss, grads, gradsak = 0.0, {},{}
        
        reg = self.reg
        N = X.shape[0]
        C = 10
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ###########################################################################
        
        x = X
        cache=[]

        regloss = 0.0
        for hd in np.arange(self.num_layers -1):
            nhw = 'W'+ str(hd+1)
            nhb = 'b'+ str(hd+1)
            W = self.params[nhw] 
            b = self.params[nhb] 
            
            a,cache_l = affine_relu_forward(x, W, b)
            x = a
            cache.append(cache_l)
            regloss += 0.5*reg*np.sum(W*W)
        
        hd = self.num_layers - 1
        nhw = 'W'+ str(hd+1)
        nhb = 'b'+ str(hd+1)
        W = self.params[nhw] 
        b = self.params[nhb] 
        
        
        scores,cache_l = affine_forward(x,W,b) 
        regloss += 0.5*reg*np.sum(W*W)



#   ---------------
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        #print("Shapes:", W1.shape, W2.shape, W3.shape, W4.shape, W5.shape)
        
        z1 = np.dot(X.reshape(N,-1),W1) + b1
        a1 = np.maximum(z1,0.0)
        
        z2 = np.dot(a1,W2) + b2
        a2 = np.maximum(z2,0.0)
        
        z3 = np.dot(a2,W3) + b3
        a3 = np.maximum(z3,0.0)
        
        z4 = np.dot(a3,W4) + b4
        a4 = np.maximum(z4,0.0)
        
        z5 = a4.dot(W5) + b5
        den = np.logaddexp.reduce(z5,axis=1)
        a5 = np.exp(z5 - den.reshape(-1,1))
        
        yoh = np.zeros((N,C))
        yoh[np.arange(N),y] = 1
        
        lossak = -1*np.sum(np.log(a5)*yoh)/N
        
        #print("Test scores:",np.sum(np.abs(scores-z5)))
        """

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, dscores = softmax_loss(scores,y)
        #print('ACG',loss,reg,regloss)
        loss += regloss
        #print("Test loss:", np.abs(loss - lossak))
        
        
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        dx,grads[nhw],grads[nhb] = affine_backward(dscores,cache_l)
        grads[nhw] += reg*self.params[nhw]
        
        """
        hd = 4
        nhw = 'W'+ str(hd)
        nhb = 'b'+ str(hd)
        dxn,grads[nhw],grads[nhb] = affine_relu_backward(dx,cache[hd-1])
        grads[nhw] += reg*self.params[nhw]
        dx = dxn
        
        
        
        
        gradz5 = (a5 - yoh)/N
        #print("Test dscores:",np.sum(np.abs(dscores-gradz5)))
        
        gradb5 = np.sum(gradz5,axis=0)  #C x 1
        gradW5 = np.dot(a4.T,gradz5)    #H x C
        
        #print(gradz5.shape,W4.T.shape)
        grada4 = np.dot(gradz5,W5.T)
        z4t = (z4>0.0).astype(np.float64)
        gradz4 = grada4*z4t
        
        gradb4 = np.sum(gradz4, axis=0)
        gradW4 = np.dot(a3.T,gradz4)
        
        gradsak['W5'] = gradW5
        gradsak['b5'] = gradb5
        gradsak['W4'] = gradW4
        gradsak['b4'] = gradb4
        """
        
        
        hdlist = np.arange(self.num_layers)
        
        
        for hd in hdlist[:0:-1]:
            nhw = 'W'+ str(hd)
            nhb = 'b'+ str(hd)
            dxn,grads[nhw],grads[nhb] = affine_relu_backward(dx,cache[hd-1])
            grads[nhw] += reg*self.params[nhw]
            dx = dxn
            
        

        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
