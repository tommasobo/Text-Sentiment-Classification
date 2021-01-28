# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:27:49 2020

BASELINE using GLOVE embeddings
Average vectors and classify

@author: Domi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection, metrics
import pickle
from scipy.sparse import *

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants


def process_tweets(tweets, em, vocab):
    '''
    process an individual file, turning the tweets into tokens, and then into average vector reps
    creates a list of averaged vector embedding numpy arrays
    line count is the number of tweets
    could also be used for test data!
    '''
    line_cnt = 0
    samples = []
    embedding_size = em.shape[1]
    print("Converting tokens to embeddings")
    print("Tweets processed:")
    for line in tweets:
        tokens = [vocab.get(t, -1) for t in line.strip().split()]
        tokens = [t for t in tokens if t >= 0] #cut out -1 words, i.e. ones not represented in the vocab / GloVe
        nr_of_words = len(tokens) 
        
        if nr_of_words == 0:
            enc = np.zeros(embedding_size) #make dummy 0 value so no test entry is lost
            samples.append(enc)
            line_cnt += 1
            if line_cnt % 100000 == 0:
                print(line_cnt)
            continue
            
        enc = np.zeros(embedding_size)
        for _, t in enumerate(tokens):
            enc += em[t, :]
        enc = enc / nr_of_words
        samples.append(enc)
        line_cnt += 1
        if line_cnt % 100000 == 0:
            print(line_cnt)
    return samples, line_cnt
        

def load_data_avg(input_data, em, vocab):
    '''
    Load our test and training data, and average out the embedded vectors
    Args:   input_data  := input data from preprocessing
            em          := embedding 'map'
            vocab       := word -> embedding map
            
    Outputs:Samples ([nr_tweets, embedding_dim])
            Labels ([nr_tweets]) (-1 or 1)
            test_samples ([nr_tweets, embedding_dim])
    '''
    pos_f = input_data.positive_tweets
    neg_f = input_data.negative_tweets
    test_f = input_data.test_tweets
    
    print("Beginning to process negative tweets")
    n_s, n_l = process_tweets(neg_f, em, vocab)
    n_s = np.array(n_s)
    print(n_s.shape)
    print("Finished processing negative tweets")
    
    print("Beginning to process positive tweets")
    p_s, p_l = process_tweets(pos_f, em, vocab)
    p_s = np.array(p_s)
    print(p_s.shape)
    print("Finished processing positive tweets")
    
    
    samples = np.concatenate((p_s, n_s))
    del p_s
    del n_s
    
    print("Processing test tweets")
    print("note: <100,000 so no counter will show")
    
    t_s, t_l = process_tweets(test_f, em, vocab)
    test_samples = np.array(t_s)
        
    print("Finished processing test tweets")
        
    labels = np.zeros((p_l + n_l, 1)) # make a label vector (all 0)
    labels[:p_l] = 1 # set the first p_l entries to 1
    labels[p_l:] = -1
    # have to shuffle this data !! (train_test_split does by default)
    return samples, labels, test_samples
                

def run_model(args, input_data):
    if args.glove_model == constants.GLOVE_PRETRAINED_TWEETS:
        with open(constants.PRETRAINED_GLOVE_EMBEDDING, 'r', encoding="utf8") as f:
            vocab = dict()
            em = np.zeros((1193513, constants.GLOVE_DIMENSIONS)) # magic number is vocab size
            # one symbol in the pretrained dictionaries is not interpreted correctly
            # it creates a token array which is too short by one value, so we must skip it
            # and adjust the index accordingly
            bad_symbol_correction = 0
            expected_size = constants.GLOVE_DIMENSIONS + 1 # + 1 as the inital token is the symbol in question
            for idx, line in enumerate(f):
                tokens = line.split()
                if len(tokens) < expected_size: 
                    bad_symbol_correction = 1
                    continue
                vocab[tokens[0]] = idx - bad_symbol_correction
                em[idx-bad_symbol_correction,:] = np.asarray(tokens[1:], "float32")
            
    else:
        vocab_fp = constants.GLOVE_VOCAB
        em_fp = constants.GLOVE_EMBEDDING
        
        if not os.path.isfile(vocab_fp):
            print("No glove_vocab.pkl file, pickling vocab")
            #Code from pickle_vocab.py from SLT course exercise
            vocab = dict()
            if not os.path.isfile(constants.GLOVE_VOCAB_TXT):
                print("glove_vocab.txt is missing from the input folder")
                print("please (re-)run data_download.sh")
                exit(1)
            with open(constants.GLOVE_VOCAB_TXT) as f:
                for idx, line in enumerate(f):
                    vocab[line.strip()] = idx
            with open(constants.GLOVE_VOCAB, 'wb') as f:
                pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
        else:
            # load our vocab list
            with open(vocab_fp, 'rb') as f:
                vocab = pickle.load(f)
                
        # if we don't already have an embedding file, create one
        if not os.path.isfile(em_fp):
            print("No embedding found, creating it")
            xs = create_embeddings(input_data, vocab)
            np.save(em_fp, xs)
        
        # load our embedding
        em = np.load(em_fp)
        
        
        
    print("Embedding shape:", em.shape)
        
    # create averages from training / test data
    print("Converting tweets to average glove vectors")
    samples, labels, test_samples = load_data_avg(input_data, em, vocab)
    
    # simple logistic regression
    regressor = linear_model.LogisticRegression(C = 1e-1)
    
    print("cross validating using logistic regressor")
    # cval
    scores = model_selection.cross_val_score(regressor, samples, labels.ravel(), cv=5)
    print("cv scores:", scores)
    
    regressor.fit(samples,labels.ravel())
    
    return regressor, regressor.predict(test_samples)
    
    
def create_embeddings(input_data, vocab):
    # CODE FROM cooc.py
    vocab_size = len(vocab)
    print("loaded vocab")
    print("vocab size:", vocab_size)
    if not os.path.isfile(constants.GLOVE_COOC):
        data, row, col = [], [], []
        counter = 1
        print("creating co-occurence matrix")
        print("number of tweets processed:")
        for fn in [constants.TRAIN_POS_DATA_SMALL, constants.TRAIN_NEG_DATA_SMALL, constants.TEST_DATA]:
            with open(fn) as f:
                for line in f:
                    tokens = [vocab.get(t, -1) for t in line.strip().split()]
                    tokens = [t for t in tokens if t >= 0]
                    for t in tokens:
                        for t2 in tokens:
                            data.append(1)
                            row.append(t)
                            col.append(t2)
        
                    if counter % 100000 == 0:
                        print(counter)
                    counter += 1
    
        cooc = coo_matrix((data, (row, col)))
        print("summing duplicates (this can take a while)")
        cooc.sum_duplicates()
        
        with open(constants.GLOVE_COOC, 'wb') as f:
            pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    # end of cooc.py code
    else:
        with open(constants.GLOVE_COOC, 'rb') as f:
            cooc = pickle.load(f)
    
    # code from glove_solution.py:
    
    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings");
    print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
    embedding_dim = constants.GLOVE_DIMENSIONS
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10
    print("Learning glove embeddings for 10 epochs")
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    
    return xs
    

def main():
    print("use main.py with arguments instead!")
    
if __name__ == '__main__':
    main()

