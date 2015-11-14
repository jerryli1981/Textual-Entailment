import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(data_folder, cv=10, clean_string=True):
    
    """
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    inputPath = './data/MovieReview/parseInput'
    with open(inputPath, 'w') as input_file:
   
        with open(pos_file, "rb") as f:
            for line in f:       
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
       
                input_file.write(orig_rev+"\n")
                 
        with open(neg_file, "rb") as f:
            for line in f:       
                rev = []
                rev.append(line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                
                input_file.write(orig_rev+"\n")
    """
    """
    import os
    parser_out = os.popen("./stanford-parser-2011-09-14/lexparser_dep.sh "+ inputPath).readlines()
    cPickle.dump(parser_out, open('./data/MovieReview/parseOutput', "wb"))
    """

    from DependencyTree import split_relation, make_tree
    parser_out = cPickle.load(open('./data/MovieReview/parseOutput',"rb"))    
    parse_text = []
    new = False
    cur_parse = []
    for line in parser_out:

        line = line.strip()

        if not line:
            new = True

        if new:
            parse_text.append(cur_parse)
            cur_parse = []
            new = False

        else:
            # print line
            rel, final_deps = split_relation(line)
            cur_parse.append( (rel, final_deps) )
    
    vocab = defaultdict(float)
    rel_dict = defaultdict(float)
    revs = []
    i=0
    for relations in parse_text:
        if i < 5331:
            label = 1
            split = np.random.randint(0,cv)
        else:
            label = 0
            split = np.random.randint(0,cv)
            
        tree = make_tree(relations)
        
        tree.label = label
        
        for node in tree.get_nodes():
            word = node.word.lower()
            if word not in vocab:
                vocab[word] += 1      
               
            for ind, rel in node.kids:
                if rel not in rel_dict:
                    if rel != "root":
                        rel_dict[rel] += 1
        i+=1
        datum  = {"label":label, 
                  "tree": tree,                             
                  "split": split}
        revs.append(datum)
   
    return revs, vocab, rel_dict
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k))            
    i = 0
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


if __name__=="__main__":    
    
    import argparse
    
    parser = argparse.ArgumentParser(description='This is a script for sentence modeling via DT-RNN')
    
    parser.add_argument('-m', '--mode', 
                        help='createDatasets, generateInputs, evaluateModel', 
                        required = True)    
      
    parser.add_argument('-w', '--w2v', 
                        help='the path of word2vector', 
                        required = False)
    
    parser.add_argument('-c', '--crossvalidation', 
                        help='the number of folds for cross validation', 
                        required = True)
    
    parser.add_argument('-n', '--numepochs',
                        help='the number of epochs for training', 
                        required = False)
    
    args= parser.parse_args()
    mode = args.mode
    w2v_file = args.w2v
    num_folds = int(args.crossvalidation)
    numepochs = int(args.numepochs)
    
    if mode == "createDatasets":
        data_folder = ["./data/MovieReview/rt-polarity.pos","./data/MovieReview/rt-polarity.neg"]    
        print "loading data...",        
        revs, vocab, rel_dict = build_data_cv(data_folder, cv=num_folds, clean_string=True)      
        print "data loaded!"
        print "number of sentences: " + str(len(revs))
        print "vocab size: " + str(len(vocab))
        print "number of relations: " + str(len(rel_dict))
        print "loading word2vec vectors...",
        w2v = load_bin_vec(w2v_file, vocab)
        print "word2vec loaded!"
        print "num words already in word2vec: " + str(len(w2v))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        cPickle.dump([revs, W, word_idx_map, vocab, rel_dict], open("./data/MovieReview/datasets_MR", "wb"))
        print "dataset created!"
        
        partitions = []
        print "Begin to generate partitons"
        for i in range(num_folds):
            partition = []    
            for rev in revs:   
                if rev["split"]==i:            
                    partition.append('Test')        
                else:  
                    partition.append('Train')
            partitions.append(partition)  
        print "Begin to dump partitions"         
        cPickle.dump(partitions, open("./data/MovieReview/partitions_MR", "wb"))  
               
    elif mode=="evaluateModel":
        from RecursiveNeuralNetwork_Architecture import evaluate_DT_RNN
    
        partitions = cPickle.load(open("./data/MovieReview/partitions_MR","rb"))
        X = cPickle.load(open("./data/MovieReview/datasets_MR","rb"))
        revs = X[0]
        W= X[1] 
        word_idx_map= X[2] 
        vocab= X[3] 
        rel_dict = X[4]  
                         
        test_results = []
        valid_results = []    
        best_test_perf = 0.
        for i in range(num_folds):
            
            partition = partitions[i];
            

            valid_perf, test_perf = evaluate_DT_RNN(revs, 
                                                    partition, 
                                                    W, 
                                                    word_idx_map, 
                                                    vocab, 
                                                    rel_dict,
                                                    batch_size = 100,
                                                    n_epochs=numepochs) 
            print ("cv: " + str(i) + ", test perf %f %%" %(test_perf * 100.))
            test_results.append(test_perf)  
            valid_results.append(valid_perf)
            if(test_perf > best_test_perf):
                best_test_perf = test_perf
               
        print('Average valid performance %f %%' %(np.mean(valid_results) * 100.))        
        print('Average test performance %f %%' %(np.mean(test_results) * 100.))    
        print('Best test performance %f %%' %(best_test_perf * 100.))
