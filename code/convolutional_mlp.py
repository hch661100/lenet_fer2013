"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time
import numpy
import pylab as pl
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from nose.plugins.skip import SkipTest
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable 
from logistic_sgd import LogisticRegression
from create_face_batch import load_data
from mlp import HiddenLayer


"""
try:
    from CrossMapNorm.python.response_norm import (
        CrossMapNorm,
        CrossMapNormUndo
    )
    from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray
    from theano.sandbox.cuda import gpu_from_host
except ImportError:
    raise SkipTest('cuda not available')
"""

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # layer_0_filter_shape: [20, 1, 5, 5] fan_in=25
        # layer_1_filter_shaoe: [50, 20, 5, 5] fan_in=500
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # layer_0_fan_out 20*5*5/4 = 125
        # layer_1_fan_out 50*5*5/4 = 312
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        # bias set to zeros
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        # W.size=(20, 1, 5, 5) (50, 20, 5, 5)
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.03, n_epochs=200,
                    dataset='../data/fer2013.dat',
                    nkerns=[32, 64], batch_size=128):
    """ Demonstrates lenet on fer2013 dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    #datasets = load_data("../data/mnist.pkl.gz")

#    train_set_x, train_set_y = datasets[0]
#    valid_set_x, valid_set_y = datasets[1]
#    test_set_x, test_set_y = datasets[2]

    train_set_x, train_set_y, test_set_x, test_set_y = load_data(dataset)

    # compute number of minibatches for training, test and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size

    img_size = 48
    c1 = 5
    p1 = 2
    l1 = (img_size - c1 + 1)/p1
    c2 = 5
    p2 = 2
    l2 = (l1 - c2 + 1)/p2
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (img_size, img_size)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,48*48)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, img_size, img_size))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (48-5+1,48-5+1)=(44,44)
    # maxpooling reduces this further to (44/2,44/2) = (22,22)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],22,22)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, img_size, img_size),
            filter_shape=(nkerns[0], 1, c1, c1), poolsize=(p1, p1))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (22-5+1,22-5+1)=(18,18)
    # maxpooling reduces this further to (18/2,18/2) = (9,9)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],9,9)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], l1, l1),
            filter_shape=(nkerns[1], nkerns[0], c2, c2), poolsize=(p2, p2))

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,64*9*9)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * l2 * l2,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=7)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    trn_err_model = theano.function([index], layer3.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_test_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    print "n_train_batches: %d"%n_train_batches
    trn_err = []
    tst_err = []
    while (epoch < n_epochs) :
        epoch = epoch + 1
        cost_ij = [train_model(i) for i in xrange(n_train_batches)]
        train_error = [trn_err_model(i) for i in xrange(n_train_batches)]
        test_error = [test_model(i) for i in xrange(n_test_batches)]

        this_train_loss = numpy.mean(cost_ij)
        this_train_error = numpy.mean(train_error)
        this_test_error = numpy.mean(test_error)

        trn_err.append(this_train_error)
        tst_err.append(this_test_error)
        x = [i for i in xrange(epoch)]
        pl.plot(x, trn_err, 'r')
        pl.plot(x, tst_err, 'g')
        pl.xlabel('epoch')
        pl.ylabel('prob_error')
        pl.savefig("prob_err.jpg") 

        print('epoch %i, cost %f train error %f %% test error %f %%' % \
                      (epoch, this_train_loss, this_train_error * 100 , this_test_error * 100.))


    end_time = time.clock()
    print('Optimization complete.')
    print('Best test score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_test_loss * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
