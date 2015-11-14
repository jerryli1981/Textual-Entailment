import sys,os
import numpy as np
import numpy
import time
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from collections import OrderedDict
from theano.sandbox import cuda

from scipy.stats import pearsonr


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape), 
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),borrow=True,name="W_conv")   
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]
        
    def predict(self, new_data, batch_size, img_shape):
        """
        predict for new data
        """
        #img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output
        


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie
    
    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie
    
    """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
    

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

    .. math::
    
    \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
    \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
    \ell (\theta=\{W,b\}, \mathcal{D})
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    
    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
    def MSE(self, y):
        data_y = [[1],[2],[3],[4],[5]]   
        data_y = theano.shared(np.asarray(data_y,
                                     dtype=theano.config.floatX),
                                     borrow=True)
        
        #y_pred_score = T.argmax(self.p_y_given_x, axis=1)
        y_pred_score = T.dot(data_y.T, self.p_y_given_x.T)
         
        return T.mean(T.sqr(y_pred_score-y))
        
    def KL_divergence(self, y):
                
        data_y = []
        
        for k in range(100):
            y_k = y[k]
            floor_y_k = T.floor(y_k)
            line = []
            for i in range(1, 6):
                if i == floor_y_k+1:
                    pi = y_k-floor_y_k
                elif i == floor_y_k:
                    pi = floor_y_k - y_k +1
                else:
                    pi = 0    
                line.append(pi)   
            data_y.append(line)       
                        
        target_distribution = theano.shared(np.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=True)
        
            
        prdicte_distribution = self.p_y_given_x
          
        return T.mean(T.sum(target_distribution * T.log(target_distribution/prdicte_distribution)))
        
class HiddenLayer(object):
    """
    Class for HiddenLayer
    """
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:            
            if activation.func_name == "ReLU":
                W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
            else:                
                W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
                                                     size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

class MLP(object):
    
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    #def __init__(self, rng, input, n_in, n_hidden, n_out, activations, use_bias=True):
    def __init__(self, rng, input, layer_sizes, activations, use_bias=True):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.activations = activations
        next_layer_input = input

        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[-2:-1]:

            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
                
        self.hiddenLayer = self.layers[-1]

        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        # Again, reuse paramters in the dropout output.
        self.logRegressionLayer = LogisticRegression(
            input=next_layer_input,
            n_in=n_in, n_out=n_out)
        
        
        self.layers.append(self.logRegressionLayer)
        
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.lgerrors = self.logRegressionLayer.errors
        
        self.KLD = self.logRegressionLayer.KL_divergence
        
        self.MSE = self.logRegressionLayer.MSE

        self.params = [ param for layer in self.layers for param in layer.params]
        
                
    def predict_label(self, new_data):
        next_layer_input = new_data
        for i,layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred 
    
    def predict_score(self, new_data):
        next_layer_input = new_data
        for i,layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        
        #data_y = [[1],[2],[3],[4],[5]]   
        data_y = [[1], [1.5], [2], [2.5], [3], [3.5], [4], [4.5], [5]]    
        data_y = theano.shared(np.asarray(data_y,
                                     dtype=theano.config.floatX),
                                     borrow=True)
        
        
        y_pred_score = T.dot(data_y.T, p_y_given_x.T)
        #y_pred_score = T.argmax(p_y_given_x, axis=1)
        return y_pred_score  
        
    def errors(self, x, y, img_h, img_w, conv_pool_layers, img_shape):
        test_pred_layers = []  

        test_layer0_input = x.reshape((x.shape[0],1,img_h, img_w))
        for conv_pool_layer in conv_pool_layers:
            test_layer0_output = conv_pool_layer.predict(test_layer0_input, x.shape[0], img_shape)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.predict_label(test_layer1_input)
        test_error = T.mean(T.neq(test_y_pred, y))

        return test_error
    
    def errorsMultiLayer(self, x, y, img_h, img_w, conv_pool_layers, img_shapes):
        test_pred_layers = []  

        test_layer1_input = x.reshape((x.shape[0],1, img_h, img_w))
              
        for i in range(0, len(conv_pool_layers), 2):
            
            conv_pool_layer1 = conv_pool_layers[i]
            img_shape0 = img_shapes[i]
            test_layer1_output = conv_pool_layer1.predict(test_layer1_input, x.shape[0], img_shape0)
            
            conv_pool_layer2 = conv_pool_layers[i+1]
            img_shape1 = img_shapes[i+1]
            test_layer2_output = conv_pool_layer2.predict(test_layer1_output, x.shape[0], img_shape1)
            test_pred_layers.append(test_layer2_output.flatten(2))
 
                        
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.predict_label(test_layer1_input)
        test_error = T.mean(T.neq(test_y_pred, y))

        return test_error
    
    def pearson(self, x, y, img_h, img_w, conv_pool_layers):
        test_pred_layers = []  
        test_layer0_input = None 

        #test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,img_h, Words.shape[1]))
        test_layer0_input = x.reshape((x.shape[0],1,img_h, img_w))
    
        for conv_pool_layer in conv_pool_layers:
            test_layer0_output = conv_pool_layer.predict(test_layer0_input, x.shape[0])
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        
        test_y_pred = self.predict_score(test_layer1_input)
        test_y_pred = test_y_pred.reshape((x.shape[0],))
        #y = y.reshape((1, x.shape[0]))
        
        return PearsonOp()(test_y_pred, y)[0] 
    
               
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
"""
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLPDropout(MLP):
    """A multilayer perceptron with dropout"""
    def __init__(self,rng,input,layer_sizes,dropout_rates,activations, use_bias=True):
        
        super(MLPDropout, self).__init__(rng=rng, 
                                         input=input, 
                                         layer_sizes=layer_sizes, 
                                         activations=activations, 
                                         use_bias=True)

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers

        self.layers = []
        self.dropout_layers = []

        next_layer_input = input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[-2:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        self.logRegressionLayer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        
        
        self.layers.append(self.logRegressionLayer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors
       
        self.dropout_KLD = self.dropout_layers[-1].KL_divergence
        self.dropout_MSE = self.dropout_layers[-1].MSE

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
def shared_dataset_float(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'float32')
    
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates
        

def evaluate_conv_net_TE(datasets,
                      partition,
                      img_h,
                      batch_size,
                      lr_decay=0.95,
                      conv_non_linear="relu",
                      dropout_rate=[0.5],
                      activations=[Iden],
                      sqr_norm_lim=9,  
                      n_epochs=200,
                      layer_sizes=[100, 10, 3], 
                      filter_hs=[3,4,5],
                      img_w = 300,
                      drop_out=False):


    rng = np.random.RandomState(23455)

    train_set, valid_set, test_set = [], [], []
    for i in range(len(datasets)):         
        if partition[i] == "Train":
            train_set.append(datasets[i])
        elif partition[i] == "Valid":
            valid_set.append(datasets[i])
        elif partition[i] == "Test":
            test_set.append(datasets[i])
            
    train_set = np.asarray(train_set)
    valid_set = np.asarray(valid_set)
    test_set = np.asarray(test_set)
    
        
    train_set_x, train_set_y = shared_dataset((train_set[:,:len(train_set[0])-3],train_set[:,-1]))
    valid_set_x, valid_set_y = shared_dataset((valid_set[:,:len(valid_set[0])-3],valid_set[:,-1]))
    test_set_x, test_set_y = shared_dataset((test_set[:,:len(test_set[0])-3],test_set[:,-1]))
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    print '... building the model'      
    feature_maps = layer_sizes[0]
    filter_shapes = []
    pool_shapes = []
    
    filter_w = img_w
           
    for filter_h in filter_hs:                     
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_shapes.append((img_h-filter_h+1, img_w-filter_w+1)) 
                                 
    conv_pool_layers = []
    layer1_inputs = []
    
    layer0_input = x.reshape((x.shape[0],1,img_h, img_w))
    
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_shape = pool_shapes[i]
        conv_pool_layer = LeNetConvPoolLayer(rng,
                                             input=layer0_input,
                                             image_shape=(batch_size, 1, img_h, img_w),
                                             filter_shape=filter_shape,
                                             poolsize=pool_shape,
                                             non_linear=conv_non_linear)
        
        layer1_input = conv_pool_layer.output.flatten(2)
        conv_pool_layers.append(conv_pool_layer)
        layer1_inputs.append(layer1_input)
      
    mlp_input = T.concatenate(layer1_inputs, 1)  
    
    layer_sizes[0] = feature_maps * len(filter_hs)
        
    classifier = None
    
    if drop_out==False:
        print 'execute mlp'
        classifier = MLP(rng=rng, 
                         input=mlp_input, 
                         layer_sizes=layer_sizes, 
                         activations=activations)
        
    elif drop_out == True:
        print 'execute mlp_dropout'
        classifier = MLPDropout(rng, 
                                input=mlp_input, 
                                layer_sizes=layer_sizes, 
                                activations=activations, 
                                dropout_rates=dropout_rate)
    
    img_shape=(batch_size, 1, img_h, img_w)
    errors = classifier.errors(x, y, img_h, img_w, conv_pool_layers, img_shape)
    #errors = classifier.lgerrors(y)
    
    test_model = theano.function(
        [index],
        errors,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        allow_input_downcast=True
    )
    
    validate_model = theano.function(
        [index],
        errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    params = classifier.params
    for conv_pool_layer in conv_pool_layers:
        params +=conv_pool_layer.params

    cost = classifier.dropout_negative_log_likelihood(y) if drop_out else classifier.negative_log_likelihood(y)
    
    updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    
    """
    new_test_set_x = np.asarray(test_set[:,:len(test_set[0])-3], dtype="float32")
    new_test_set_y = np.asarray(test_set[:,-3], dtype="int32")
    ids = np.asarray(test_set[:,-3],dtype="int32")
    
    test_pred_layers = []  
    test_size = new_test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h, Words.shape[1]))
    for conv_pool_layer in conv_pool_layers:
        test_layer0_output = conv_pool_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)    
    test_y_pred = classifier.predict_label(test_layer1_input)
    test_model_output = theano.function([x], test_y_pred)
    
      
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error) 
    """
    
    #print 'Used the','gpu' if np.any([isinstance(x.op, cuda.GpuElemwise) for x in train_model.maker.fgraph.toposort()])  else 'cpu'

    print '... training'    
    best_validation_loss = np.inf
    best_test_lost = np.inf
    start_time = time.clock()
    shuffle_batch = True
    epoch = 0

    while epoch < n_epochs:
        epoch = epoch + 1
        print 'epoch @ epoch = ', epoch
        
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                #print minibatch_index
                cost_ij = train_model(minibatch_index) 
                #print cost_ij

        else:
            for minibatch_index in xrange(n_train_batches):
                #print minibatch_index
                cost_ij = train_model(minibatch_index) 
                #print cost_ij 

        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        print('epoch %i, validation error %f %%' %
                      (epoch, this_validation_loss * 100.))

        # if we got the best validation score until now
        if this_validation_loss <= best_validation_loss :
        #if True:

            best_validation_loss = this_validation_loss           
            print('epoch %i, best validation loss %f %%' %
                      (epoch, best_validation_loss * 100.)) 
           
            test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
            test_loss = np.mean(test_losses)            

            """
            test_y_pred = test_model_output(new_test_set_x)
            f = open("TE_predict", "wb")
            f.write("pair_ID"+"\t"+"entailment_judgment"+"\t"+"relatedness_score"+"\n")
            l = test_y_pred
            i = 0
            for ele in l:
                if ele == 0:
                    f.write(str(ids[i]) +"\t" + "NEUTRAL"+"\t"+"NA" + "\n")
                elif ele ==1:
                    f.write(str(ids[i]) +"\t" + "CONTRADICTION"+"\t"+"NA" + "\n")
                elif ele == 2:
                    f.write(str(ids[i]) +"\t" + "ENTAILMENT"+"\t"+"NA" + "\n")
                i += 1
            f.close()

            ro.r('ifile="/home/lipeng/DCNN/TE_predict"')
            ro.r('gold = "/home/lipeng/DCNN/SICK_test_annotated.txt"')
            ro.r('read.delim(ifile, sep="\t", header=T, stringsAsFactors=F) -> score')
            ro.r('read.delim(gold, sep="\t", header=T) -> gold')
            ro.r('score <- score[order(score$pair_ID), ]')
            ro.r('gold <- gold[order(gold$pair_ID), ]')
            ro.r('accuracy <- sum(score$entailment_judgment == gold$entailment_judgment) / length(score$entailment_judgment)*100')
            accuracy = ro.r('accuracy')

            print "Test Error by R on all data is " + str(1- float(str(accuracy).split()[1])/100)
            """
            
            """
            test_loss_all = test_model_all(new_test_set_x,new_test_set_y) 
            print( "Test error on all data by python is " + str(test_loss_all))
            """
            
            print('epoch %i, test error of best model %f %%' % (epoch,  test_loss * 100.) )
            
            if test_loss < best_test_lost:
                best_test_lost = test_loss
                                                       
    end_time = time.clock()

    print('Best validation loss of %f %% , '
          'with test loss %f %%' %
          (best_validation_loss * 100., best_test_lost * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    return 1-best_validation_loss, 1-best_test_lost

def evaluate_conv_net_TE_2(datasets,
                      partition,
                      img_h,
                      batch_size,
                      lr_decay=0.95,
                      conv_non_linear="relu",
                      dropout_rate=[0.5],
                      activations=[Iden, Iden],
                      sqr_norm_lim=9,  
                      n_epochs=200,
                      layer_sizes=[100, 10, 3], 
                      filter_hs=[3,4,5],
                      img_w = 300,
                      drop_out=False):

    original_img_h = img_h
    original_img_w = img_w
    rng = np.random.RandomState(23455)

    train_set, valid_set, test_set = [], [], []
    for i in range(len(datasets)):         
        if partition[i] == "Train":
            train_set.append(datasets[i])
        elif partition[i] == "Valid":
            valid_set.append(datasets[i])
        elif partition[i] == "Test":
            test_set.append(datasets[i])
            
    train_set = np.asarray(train_set)
    valid_set = np.asarray(valid_set)
    test_set = np.asarray(test_set)
        
    train_set_x, train_set_y = shared_dataset((train_set[:,:len(train_set[0])-3],train_set[:,-1]))
    valid_set_x, valid_set_y = shared_dataset((valid_set[:,:len(valid_set[0])-3],valid_set[:,-1]))
    test_set_x, test_set_y = shared_dataset((test_set[:,:len(test_set[0])-3],test_set[:,-1]))
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    print '... building the model'      
    feature_maps_1 = layer_sizes[0]
    filter_shapes_1 = []
    pool_shapes_1 = []
    
    filter_w = img_w/4
    
    for filter_h in filter_hs:                     
        filter_shapes_1.append((feature_maps_1, 1, filter_h, filter_w))
        pool_shapes_1.append((2, 2)) 
        
        
    feature_maps_2 = layer_sizes[1]
    filter_shapes_2 = []
    pool_shapes_2 = []
        
    for filter_h in filter_hs:                     
        filter_shapes_2.append((feature_maps_2, feature_maps_1, filter_h, filter_w))
        pool_shapes_2.append((2, 2)) 
    
                                    
    conv_pool_layers = []
    img_shapes = []
   
    layer0_input = x.reshape((x.shape[0],1,original_img_h, original_img_w))
    
    layer2_inputs = []
    totalUnits = 0
    for i in xrange(len(filter_hs)):
        filter_shape_1 = filter_shapes_1[i]
        pool_shape_1 = pool_shapes_1[i]
        conv_pool_layer1 = LeNetConvPoolLayer(rng,
                                             input=layer0_input,
                                             image_shape=(batch_size, 1, original_img_h, original_img_w),
                                             filter_shape=filter_shape_1,
                                             poolsize=pool_shape_1,
                                             non_linear=conv_non_linear)
        
        
        layer1_input = conv_pool_layer1.output
        conv_pool_layers.append(conv_pool_layer1)
        img_shapes.append((batch_size, 1, original_img_h, original_img_w))
       
        filter_h_1 = filter_shape_1[2]
        filter_w_1 = filter_shape_1[3]
        pool_h_1 = pool_shape_1[0]
        pool_w_1 = pool_shape_1[1]
        
        
        
      
        filter_shape_2 = filter_shapes_2[i]
        img_shape = (batch_size, filter_shape_2[1], 
                     (original_img_h-filter_h_1+1)/pool_h_1, (original_img_w-filter_w_1+1)/pool_w_1)
        img_shapes.append(img_shape)
        
        
        pool_shape_2 = pool_shapes_2[i]
        conv_pool_layer2 = LeNetConvPoolLayer(rng,
                                             input=layer1_input,
                                             image_shape=img_shape,
                                             filter_shape=filter_shape_2,
                                             poolsize=pool_shape_2,
                                             non_linear=conv_non_linear)
        
        layer2_input = conv_pool_layer2.output.flatten(2)
        conv_pool_layers.append(conv_pool_layer2)
       
        layer2_inputs.append(layer2_input)   
        
        filter_h_2 = filter_shape_2[2]
        filter_w_2 = filter_shape_2[3]
        pool_h_2 = pool_shape_2[0]
        pool_w_2 = pool_shape_2[1]
        
        
        img_h = ((original_img_h-filter_h_1+1)/pool_h_1 - filter_h_2 +1)/pool_h_2
        img_w = ((original_img_w-filter_w_1+1)/pool_w_1 - filter_w_2 +1)/pool_w_2
        totalUnits += feature_maps_2 * img_h * img_w
        
            
    mlp_input = T.concatenate(layer2_inputs, 1) 
    

    layer_sizes[-3] = totalUnits
        
    classifier = None
    
    if drop_out==False:
        print 'execute mlp'
        classifier = MLP(rng=rng, 
                         input=mlp_input, 
                         layer_sizes=layer_sizes, 
                         activations=activations)
        
    elif drop_out == True:
        print 'execute mlp_dropout'
        classifier = MLPDropout(rng, 
                                input=mlp_input, 
                                layer_sizes=layer_sizes, 
                                activations=activations, 
                                dropout_rates=dropout_rate)
       
    errors = classifier.errorsMultiLayer(x, y, original_img_h, original_img_w, conv_pool_layers, img_shapes)
    #errors = classifier.lgerrors(y)
    
    test_model = theano.function(
        [index],
        errors,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        allow_input_downcast=True
    )
    
    validate_model = theano.function(
        [index],
        errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    params = classifier.params
    for conv_pool_layer in conv_pool_layers:
        params +=conv_pool_layer.params

    cost = classifier.dropout_negative_log_likelihood(y) if drop_out else classifier.negative_log_likelihood(y)
    
    updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    
    """
    new_test_set_x = np.asarray(test_set[:,:len(test_set[0])-3], dtype="float32")
    new_test_set_y = np.asarray(test_set[:,-3], dtype="int32")
    ids = np.asarray(test_set[:,-3],dtype="int32")
    
    test_pred_layers = []  
    test_size = new_test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h, Words.shape[1]))
    for conv_pool_layer in conv_pool_layers:
        test_layer0_output = conv_pool_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)    
    test_y_pred = classifier.predict_label(test_layer1_input)
    test_model_output = theano.function([x], test_y_pred)
    
      
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error) 
    """
    
    #print 'Used the','gpu' if np.any([isinstance(x.op, cuda.GpuElemwise) for x in train_model.maker.fgraph.toposort()])  else 'cpu'

    print '... training'    
    best_validation_loss = np.inf
    best_test_lost = np.inf
    start_time = time.clock()
    shuffle_batch = True
    epoch = 0

    while epoch < n_epochs:
        epoch = epoch + 1
        print 'epoch @ epoch = ', epoch
        
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                #print minibatch_index
                cost_ij = train_model(minibatch_index) 
                #print cost_ij

        else:
            for minibatch_index in xrange(n_train_batches):
                #print minibatch_index
                cost_ij = train_model(minibatch_index) 
                #print cost_ij 

        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        print('epoch %i, validation error %f %%' %
                      (epoch, this_validation_loss * 100.))

        # if we got the best validation score until now
        if this_validation_loss <= best_validation_loss:
        #if True:

            best_validation_loss = this_validation_loss           
            print('epoch %i, best validation loss %f %%' %
                      (epoch, best_validation_loss * 100.)) 
           
            test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
            test_loss = np.mean(test_losses)            

            """
            test_y_pred = test_model_output(new_test_set_x)
            f = open("TE_predict", "wb")
            f.write("pair_ID"+"\t"+"entailment_judgment"+"\t"+"relatedness_score"+"\n")
            l = test_y_pred
            i = 0
            for ele in l:
                if ele == 0:
                    f.write(str(ids[i]) +"\t" + "NEUTRAL"+"\t"+"NA" + "\n")
                elif ele ==1:
                    f.write(str(ids[i]) +"\t" + "CONTRADICTION"+"\t"+"NA" + "\n")
                elif ele == 2:
                    f.write(str(ids[i]) +"\t" + "ENTAILMENT"+"\t"+"NA" + "\n")
                i += 1
            f.close()

            ro.r('ifile="/home/lipeng/DCNN/TE_predict"')
            ro.r('gold = "/home/lipeng/DCNN/SICK_test_annotated.txt"')
            ro.r('read.delim(ifile, sep="\t", header=T, stringsAsFactors=F) -> score')
            ro.r('read.delim(gold, sep="\t", header=T) -> gold')
            ro.r('score <- score[order(score$pair_ID), ]')
            ro.r('gold <- gold[order(gold$pair_ID), ]')
            ro.r('accuracy <- sum(score$entailment_judgment == gold$entailment_judgment) / length(score$entailment_judgment)*100')
            accuracy = ro.r('accuracy')

            print "Test Error by R on all data is " + str(1- float(str(accuracy).split()[1])/100)
            """
            
            """
            test_loss_all = test_model_all(new_test_set_x,new_test_set_y) 
            print( "Test error on all data by python is " + str(test_loss_all))
            """
            
            print('epoch %i, test error of best model %f %%' % (epoch,  test_loss * 100.) )
            
            if test_loss <= best_test_lost:
                best_test_lost = test_loss
                                                       
    end_time = time.clock()

    print('Best validation loss of %f %% , '
          'with test loss %f %%' %
          (best_validation_loss * 100., best_test_lost * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    return 1-best_validation_loss, 1-best_test_lost

def evaluate_conv_net_SR(datasets,
                      partition,
                      img_h,
                      batch_size,
                      lr_decay=0.95,
                      conv_non_linear="relu",
                      dropout_rate=[0.5],
                      activations=[Iden],
                      sqr_norm_lim=9,  
                      n_epochs=200,
                      layer_sizes=[100, 10, 3], 
                      filter_hs=[3,4,5],
                      img_w = 300,
                      drop_out=False):
    
    #from Scipy_Theano_Wrapper import PearsonOp
    import rpy2.robjects as ro

    rng = np.random.RandomState(23455)

    train_set, valid_set, test_set = [], [], []
    for i in range(len(datasets)):         
        if partition[i] == "Train":
            train_set.append(datasets[i])
        elif partition[i] == "Valid":
            valid_set.append(datasets[i])
        elif partition[i] == "Test":
            test_set.append(datasets[i])
            
    scale = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]    

    for ele in train_set:
        score = ele[-2]
        idx = -1
        min = 10
        for i, r in enumerate(scale):
            if abs(score-r) < min:
                min = abs(score-r)
                idx = i
        ele[-2] = idx
    
    for ele in valid_set:
        score = ele[-2]
        idx = -1
        min = 10
        for i, r in enumerate(scale):
            if abs(score-r) < min:
                min = abs(score-r)
                idx = i
        ele[-2] = idx

    
    train_set = np.asarray(train_set)  
    valid_set = np.asarray(valid_set)    
    test_set = np.asarray(test_set)
       
    #train_set_x, train_set_y = shared_dataset((train_set[:,:len(train_set[0])-3],np.rint(train_set[:,-2])-1))
    #valid_set_x, valid_set_y = shared_dataset((valid_set[:,:len(valid_set[0])-3],np.rint(valid_set[:,-2])-1))
    
    train_set_x, train_set_y = shared_dataset((train_set[:,:len(train_set[0])-3],train_set[:,-2]))
    valid_set_x, valid_set_y = shared_dataset((valid_set[:,:len(valid_set[0])-3],valid_set[:,-2]))
    
    
    test_set_x, test_set_y = shared_dataset_float((test_set[:,:len(test_set[0])-3],test_set[:,-2]))
    

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
                        
     
    print '... building the model'       
    feature_maps = layer_sizes[0]
    filter_shapes = []
    pool_shapes = []
    
    filter_w = img_w            
    for filter_h in filter_hs:                      
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_shapes.append((img_h-filter_h+1, img_w-filter_w+1)) 
            
            
    conv_pool_layers = []
    layer1_inputs = []
    
    layer0_input = x.reshape((x.shape[0],1,img_h, img_w))
    
    
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_shape = pool_shapes[i]
        
        
        conv_pool_layer = LeNetConvPoolLayer(rng,
                                             input=layer0_input,
                                             image_shape=(batch_size, 1, img_h, img_w),
                                             filter_shape=filter_shape,
                                             poolsize=pool_shape,
                                             non_linear=conv_non_linear)
        
               
        layer1_input = conv_pool_layer.output.flatten(2)
        conv_pool_layers.append(conv_pool_layer)
        layer1_inputs.append(layer1_input)
      
    mlp_input = T.concatenate(layer1_inputs, 1)  
    
    layer_sizes[0] = feature_maps * len(filter_hs)
        
   
    if drop_out==False:
        print 'execute mlp'
        classifier = MLP(rng=rng, 
                         input=mlp_input, 
                         layer_sizes=layer_sizes, 
                         activations=activations)
        
    elif drop_out == True:
        print 'execute mlp_dropout'
        classifier = MLPDropout(rng, 
                                input=mlp_input, 
                                layer_sizes=layer_sizes, 
                                activations=activations, 
                                dropout_rates=dropout_rate)
    

    """
    pearson_score = classifier.pearson(x, y, Words, img_h, img_w, conv_pool_layers)      
    test_model = theano.function(
        [index],
        pearson_score,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    """
  
    #cost = classifier.dropout_KLD(y) if drop_out else classifier.KLD(y)  
    #mse = classifier.dropout_MSE(y) if drop_out else classifier.MSE(y)  
    errors = classifier.errors(x, y, img_h, img_w, conv_pool_layers, (batch_size, 1, img_h, img_w))
    cost = classifier.dropout_negative_log_likelihood(y) if drop_out else classifier.negative_log_likelihood(y)
    
    validate_model = theano.function(
        [index],
        errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    params = classifier.params
    for conv_pool_layer in conv_pool_layers:
        params +=conv_pool_layer.params

    updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    new_test_set_x = np.asarray(test_set[:,:len(test_set[0])-3], dtype="float32")
    new_test_set_y = np.asarray(test_set[:,-2], dtype="float32")
    ids = np.asarray(test_set[:,-3],dtype="int32")
    
    test_size = new_test_set_x.shape[0]
    img_shape=(test_size, 1, img_h, img_w)
    
    test_pred_layers = []  
    
    #test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h, Words.shape[1]))
    test_layer0_input = x.reshape((test_size,1,img_h, img_w))
    for conv_pool_layer in conv_pool_layers:
        test_layer0_output = conv_pool_layer.predict(test_layer0_input, test_size, img_shape)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)    
    test_y_pred = classifier.predict_score(test_layer1_input)
    test_model_output = theano.function([x], test_y_pred)
    
    
    """
    test_y_pred_1 = test_y_pred.reshape((x.shape[0],))
    test_correlation = PearsonOp()(test_y_pred_1, y)[0]
    test_model_all = theano.function([x,y], test_correlation) 
    """
    
    
    #print 'Used the','gpu' if np.any([isinstance(x.op, cuda.GpuElemwise) for x in train_model.maker.fgraph.toposort()])  else 'cpu'
    
    print '... training'    
    best_validation_mse = np.inf
    best_test_correlation = 0.
    start_time = time.clock()
    shuffle_batch = True
    epoch = 0
    
    

    while epoch < n_epochs:
        epoch = epoch + 1
        print 'epoch @ epoch = ', epoch
        
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                #print minibatch_index
                cost_ij = train_model(minibatch_index) 
                #print cost_ij

        else:
            for minibatch_index in xrange(n_train_batches):
                #print minibatch_index
                cost_ij = train_model(minibatch_index) 
                #print cost_ij 

        # compute zero-one loss on validation set
        
        validation_mses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_mse = np.mean(validation_mses)

        if this_validation_mse <= best_validation_mse or True:   
            best_validation_mse = this_validation_mse           
            print('epoch %i, best validation mse %f %%' %
                      (epoch, best_validation_mse * 100.)) 


            test_y_pred = test_model_output(new_test_set_x)
            
            
            test_correlation = pearsonr(test_y_pred[0],new_test_set_y)[0]
            print('epoch %i, test correlation of best model %f %%' % (epoch,  test_correlation * 100.) )
            
            
            """
            with open("SR_predict", "wb") as f:
                f.write("pair_ID"+"\t"+"entailment_judgment" +"\t"+"relatedness_score"+"\n")
                for i, ele in enumerate(test_y_pred[0]):
                    f.write(str(ids[i])+"\t"+"NA"+"\t"+str(ele)+'\n')
                    
            with open("SR_gold", "wb") as f:
                f.write("pair_ID"+"\t"+"entailment_judgment" +"\t"+"relatedness_score"+"\n")
                for i, ele in enumerate(new_test_set_y):
                    f.write(str(ids[i])+"\t"+"NA"+"\t"+str(ele)+'\n')
                        
            ro.r('ifile = "SR_predict"')
            ro.r('gold = "SR_gold"')
            ro.r('read.delim(ifile, sep="\t", header=T, stringsAsFactors=F) -> score')
            ro.r('read.delim(gold, sep="\t", header=T) -> gold')
            ro.r('score <- score[order(score$pair_ID), ]')
            ro.r('gold <- gold[order(gold$pair_ID), ]')
            ro.r('pearson <- cor(score$relatedness_score, gold$relatedness_score)')
            pearson = ro.r('pearson')
            #print "Pearson correlation is " + str(pearson).split()[1]
            #ro.r('spearman <- cor(score$relatedness_score, gold$relatedness_score, method = "spearman")')
            #spearman = ro.r('spearman')
            #print "Spearman correlation is " + str(spearman).split()[1]
            #ro.r('MSE <- sum((score$relatedness_score - gold$relatedness_score)^2) / length(score$relatedness_score)')
            #MSE = ro.r('MSE')
            #print "MSE is " + str(MSE).split()[1]
                       
            if str(pearson).split()[1] != "NA":
                print('epoch %i, test correlation of best model %f %%' % (epoch,  float(str(pearson).split()[1]) * 100.) )
                test_correlation = float(str(pearson).split()[1])  
                if test_correlation > best_test_correlation:
                    best_test_correlation = test_correlation
            """   
            
            if test_correlation > best_test_correlation:
                best_test_correlation = test_correlation
            
        
    end_time = time.clock()

    print('Best validation correlation of %f %% , '
          'with test correlation %f %%' %
          (best_validation_mse * 100., best_test_correlation * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    return best_validation_mse, best_test_correlation

