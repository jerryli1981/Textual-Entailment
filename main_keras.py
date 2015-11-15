import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import dependency_tree as tr

sys.path.insert(0, os.path.abspath('../keras'))

import keras

def load_data(data, wordEmbeddings, args, maxlen=36):

    
    X1 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)
    X2 = np.zeros((len(data), maxlen, args.wvecDim), dtype=np.float32)

    Y_scores = np.zeros((len(data)), dtype=np.float32)

    labels = []
    for i, (label, score, l_tree, r_tree) in enumerate(data):

        for j, Node in enumerate(l_tree.nodes):
            X1[i, j] =  wordEmbeddings[:, Node.index]

        for k, Node in enumerate(r_tree.nodes):
            X2[i, k] =  wordEmbeddings[:, Node.index]

        Y_scores[i] = score
        labels.append(label)

    labels = np.asarray(labels, dtype='int32')

    Y_labels = np.zeros((len(labels), args.numLabels))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.
        
    return X1, X2, Y_labels, Y_scores

def build_network(args, maxlen=36):

    input_shape=(maxlen, args.wvecDim)

    print("Building model and compiling functions...")

    l_lstm_1 = keras.models.Sequential()
    l_lstm_1.add(keras.layers.recurrent.LSTM(output_dim=args.wvecDim, 
        return_sequences=True, input_shape=input_shape))
    #l_lstm_1.add(keras.layers.core.Flatten())

    l_lstm_2 = keras.models.Sequential()
    l_lstm_2.add(keras.layers.recurrent.LSTM(output_dim=args.wvecDim, 
        return_sequences=True, input_shape=input_shape))
    #l_lstm_2.add(keras.layers.core.Flatten())

    l_mul = keras.models.Sequential()
    l_mul.add(keras.layers.core.Merge([l_lstm_1, l_lstm_2], mode='mul'))
    #l_mul.add(keras.layers.core.Dense(output_dim=args.wvecDim))

    l_sub = keras.models.Sequential()
    l_sub.add(keras.layers.core.Merge([l_lstm_1, l_lstm_2], mode='abs_sub'))
    #l_sub.add(keras.layers.core.Dense(output_dim=args.wvecDim))

    model = keras.models.Sequential()
    model.add(keras.layers.core.Merge([l_mul, l_sub], mode='sum'))
    model.add(keras.layers.core.Reshape((1, maxlen, args.wvecDim)))


    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model.add(keras.layers.convolutional.Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, maxlen, args.wvecDim)))

    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.convolutional.Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(keras.layers.core.Dropout(0.25))

    model.add(keras.layers.core.Flatten())
    model.add(keras.layers.core.Dense(128))
    model.add(keras.layers.core.Activation('relu'))
    model.add(keras.layers.core.Dropout(0.5))
    model.add(keras.layers.core.Dense(args.numLabels, init='uniform'))
    model.add(keras.layers.core.Activation('softmax'))

    #rms = RMSprop()
    #sgd = SGD(lr=0.1, decay=1e-6, mementum=0.9, nesterov=True)
    adagrad = keras.optimizers.Adagrad(args.step)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)

    train_fn = model.train_on_batch
    test_fn = model.test_on_batch 
    return train_fn, test_fn

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
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=10)
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


    # Create neural network model (depending on first command line parameter)
    #train_fn, val_fn = build_network_lasagne(args, input1_var, input2_var)
    train_fn, test_fn= build_network(args)

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
            train_err += train_fn([inputs1, inputs2], labels)
            train_batches += 1

        # And a full pass over the validation data:
        
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X1_dev, X2_dev, Y_labels_dev, Y_scores_dev, 500, shuffle=False):
            inputs1, inputs2, labels, _ = batch
            err, acc = test_fn([inputs1, inputs2], labels, accuracy=True)
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
        err, acc = test_fn([inputs1, inputs2], labels, accuracy=True)
        test_err += err
        test_acc += acc
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  Best validate accuracy:\t\t{:.2f} %".format(best_val_acc))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

