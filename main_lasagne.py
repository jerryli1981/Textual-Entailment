import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import dependency_tree as tr

sys.path.insert(0, os.path.abspath('../Lasagne'))
#sys.path.append('../Lasagne')

import lasagne

def load_data_matrix(data, args, seq_len=36, n_children=6, unfinished_flag=-2):

    Y = np.zeros((len(data), args.outputDim+1), dtype=np.float32)
    scores = np.zeros((len(data)), dtype=np.float32)

    # to store hidden representation
    #(rootFlag, finishedFlag, globalgovIdx, n_children* (locaDepIdx, globalDepIdx, relIdx) , hiddenRep)
    storage_dim = 1 + 1 + 1 + 3*n_children + args.wvecDim

    X1 = np.zeros((len(data), seq_len, storage_dim), dtype=np.float32)
    X1.fill(-1.0)
    X2 = np.zeros((len(data), seq_len, storage_dim), dtype=np.float32)
    X2.fill(-1.0)
    
    for i, (score, item) in enumerate(data):
        first_t, second_t= item

        sim = score
        ceil = np.ceil(sim)
        floor = np.floor(sim)
        if ceil == floor:
            Y[i, floor] = 1
        else:
            Y[i, floor] = ceil-sim
            Y[i, ceil] = sim-floor

        f_idxSet = set()
        for govIdx, depIdx in first_t.dependencies:
            f_idxSet.add(govIdx)
            f_idxSet.add(depIdx)

        for j, Node in enumerate(first_t.nodes):

            if j not in f_idxSet:
                continue

            node_vec = np.zeros((storage_dim,), dtype=np.float32)
            node_vec.fill(-1.0)
            if j == first_t.rootIdx:
                node_vec[0] = 1

            node_vec[1] = unfinished_flag
            node_vec[2] = Node.index

            if len(Node.kids) != 0:

                r = range(0, 3*n_children, 3)
                r = r[:len(Node.kids)]
                for d, c in enumerate(r):
                    localDepIdx, rel = Node.kids[d]
                    node_vec[3+c] = localDepIdx
                    node_vec[4+c] = first_t.nodes[localDepIdx].index
                    node_vec[5+c] = rel.index


            X1[i, j] = node_vec


        s_idxSet = set()
        for govIdx, depIdx in second_t.dependencies:
            s_idxSet.add(govIdx)
            s_idxSet.add(depIdx)

        for j, Node in enumerate(second_t.nodes):

            if j not in s_idxSet:
                continue

            node_vec = np.zeros((storage_dim,), dtype=np.float32)
            node_vec.fill(-1.0)
            if j == second_t.rootIdx:
                node_vec[0] = 1

            node_vec[1] = unfinished_flag
            node_vec[2] = Node.index

            if len(Node.kids) != 0:

                r = range(0, 3*n_children, 3)
                r = r[:len(Node.kids)]
                for d, c in enumerate(r):
                    localDepIdx, rel = Node.kids[d]
                    node_vec[3+c] = localDepIdx
                    node_vec[4+c] = second_t.nodes[localDepIdx].index
                    node_vec[5+c] = rel.index

            X2[i, j] = node_vec
   
        scores[i] = score

    Y = Y[:, 1:]

    input_shape = (len(data), seq_len, storage_dim)
      
    return X1, X2, Y, scores, input_shape

def load_data(data, wordEmbeddings, args, maxlen=36):

    
    X1 = np.zeros((len(data), 1, maxlen, args.wvecDim), dtype=np.float32)
    X2 = np.zeros((len(data), 1, maxlen, args.wvecDim), dtype=np.float32)

    Y_scores = np.zeros((len(data)), dtype=np.float32)

    Y_labels = np.zeros((len(data)), dtype=np.int32)
    for i, (label, score, l_tree, r_tree) in enumerate(data):

        for j, Node in enumerate(l_tree.nodes):
            X1[i, 0, j] =  wordEmbeddings[:, Node.index]

        for k, Node in enumerate(r_tree.nodes):
            X2[i, 0, k] =  wordEmbeddings[:, Node.index]

        Y_scores[i] = score
        Y_labels[i] = label
        
    return X1, X2, Y_labels, Y_scores

def build_network(args, input1_var=None, input2_var=None, maxlen=36):

    print("Building model and compiling functions...")

    """
    1. for each sentence, first do LSTM
    sent_1: (None, maxlen, wvecDim)
    sent_2: (None, maxlen, wvecDim)

    2. for each lstm output, do feature mean pooling
    sent_1: (None, maxlen, wvecDim/4)
    sent_2: (None, maxlen, wvecDim/4)

    2. Do multiply and abs_sub.
    mul: (None, maxlen, wvecDim/4)
    sub: (None, maxlen, wvecDim/4)

    3. hs= sigmoid(W1 * mul + W2* sub)

    4. pred = softmax(W*hs + b)

    """

    l1_in = lasagne.layers.InputLayer(shape=(None, 1, maxlen, args.wvecDim),
                                     input_var=input1_var)

    l2_in = lasagne.layers.InputLayer(shape=(None, 1, maxlen, args.wvecDim),
                                     input_var=input2_var)

    """
    GRAD_CLIP = args.wvecDim/2
    l_forward_1 = lasagne.layers.LSTMLayer(
        l1_in, args.wvecDim, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)


    l_forward_2 = lasagne.layers.LSTMLayer(
        l2_in, args.wvecDim, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    """

    l12_mul = lasagne.layers.ElemwiseMergeLayer([l1_in, l2_in], merge_function=T.mul)
    l12_sub = lasagne.layers.ElemwiseMergeLayer([l1_in, l2_in], merge_function=T.sub)
    l12_sub = lasagne.layers.AbsLayer(l12_sub)

    #l12_mul_Dense = lasagne.layers.DenseLayer(l12_mul, num_units=args.hiddenDim, nonlinearity=None, b=None)

    #l12_sub_Dense = lasagne.layers.DenseLayer(l12_sub, num_units=args.hiddenDim, nonlinearity=None, b=None)

    joined = lasagne.layers.ElemwiseSumLayer([l12_mul, l12_sub])


    con2D = lasagne.layers.Conv2DLayer(
            joined, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    maxPool = lasagne.layers.MaxPool2DLayer(con2D, pool_size=(2, 2))


    if args.mlpActivation == "sigmoid":
        act_fun = lasagne.nonlinearities.sigmoid
    elif args.mlpActivation == "rectify":
        act_fun = lasagne.nonlinearities.rectify
    else:
        raise "Need set mlp activation correctly"

    l_hid1 = lasagne.layers.NonlinearityLayer(maxPool, nonlinearity=act_fun)

    #l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=args.numLabels,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def iterate_minibatches(inputs1, inputs2, Y_labels, Y_scores, batchsize, shuffle=False):
    assert len(inputs1) == len(Y_labels)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs1[excerpt], inputs2[excerpt], Y_labels[excerpt], Y_scores[excerpt]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="sgd")
    parser.add_argument("--epochs",dest="epochs",type=int,default=50)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--rangeScores",dest="rangeScores",type=int,default=5)
    parser.add_argument("--numLabels",dest="numLabels",type=int,default=3)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    parser.add_argument("--wvecDim",dest="wvecDim",type=int,default=30)
    parser.add_argument("--outFile",dest="outFile",type=str, default="models/test.bin")
    parser.add_argument("--mlpActivation",dest="mlpActivation",type=str,default="sigmoid")
    args = parser.parse_args()
         
    trainTrees = tr.loadTrees("train")
    devTrees = tr.loadTrees("dev")
    testTrees = tr.loadTrees("test")
    
    print "train number %d"%len(trainTrees)
    print "dev number %d"%len(devTrees)
    print "test number %d"%len(testTrees)

    wordEmbeddings = tr.loadWord2VecMap()[:args.wvecDim, :]
    
    X1_train, X2_train, Y_labels_train, Y_scores_train = load_data(trainTrees, wordEmbeddings, args)
    X1_dev, X2_dev, Y_labels_dev, Y_scores_dev = load_data(devTrees, wordEmbeddings, args)
    X1_test, X2_test, Y_labels_test, Y_scores_test = load_data(testTrees, wordEmbeddings, args)

    # Prepare Theano variables for inputs and targets
    input1_var = T.tensor4('inputs_1')
    input2_var = T.tensor4('inputs_2')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    network = build_network(args, input1_var, input2_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)

    #updates = lasagne.updates.nesterov_momentum(
            #loss, params, learning_rate=0.01, momentum=0.9)

    if args.optimizer == "sgd":
        updates = lasagne.updates.sgd(loss, params, args.step)
    elif args.optimizer == "adagrad":
        updates = lasagne.updates.adagrad(loss, params, args.step)
    elif args.optimizer == "adadelta":
        updates = lasagne.updates.adadelta(loss, params, args.step)
    else:
        raise "Need set optimizer correctly"
 
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input1_var, input2_var, target_var], loss, updates=updates, allow_input_downcast=True)

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input1_var, input2_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    best_val_acc = 0
    for epoch in range(args.epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X1_train, X2_train, Y_labels_train, Y_scores_train, args.minibatch, shuffle=True):
            inputs1, inputs2, labels, _ = batch
            train_err += train_fn(inputs1, inputs2, labels)
            train_batches += 1

        # And a full pass over the validation data:
        
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X1_dev, X2_dev, Y_labels_dev, Y_scores_dev, 500, shuffle=False):
            inputs1, inputs2, labels, _ = batch
            err, acc = val_fn(inputs1, inputs2, labels)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


        val_score = val_acc / val_batches * 100
        print("  validation accuracy:\t\t{:.2f} %".format(val_score))
        if best_val_acc < val_score:
            best_val_acc = val_score

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X1_test, X2_test, Y_labels_test, Y_scores_test, 4500, shuffle=False):
        inputs1, inputs2, targets, scores = batch
        err, acc = val_fn(inputs1, inputs2, targets)
        test_err += err
        test_acc += acc
        test_batches += 1



    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  Best validate accuracy:\t\t{:.2f} %".format(best_val_acc))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
