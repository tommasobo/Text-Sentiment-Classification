import csv
import sys
import os
import argparse
import numpy as np
import pandas as pd

os.environ['TF_KERAS'] = '1'

import tensorflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '../../src/')
import constants
import preprocessing
import pickle
from preprocessing import InputData
from random import choice

import keras_bert
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX
from keras_bert import get_custom_objects

import re, os
import codecs

from tensorflow.python import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData, preprocess_single_tweet
from tensorflow.python.ops.math_ops import erf, sqrt

maxlen = 55
config_path = '../data/input/uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '../data/input/uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '../data/input/uncased_L-24_H-1024_A-16/vocab.txt'

def predict_bert(args, input_data, load_path):

    BATCH_SIZE = args.batch_size
    EPOCHS = args.number_epochs
    STEPS = args.step_size

    # load tweets 
    input_data.positive_tweets = preprocess_single_tweet(input_data.positive_tweets)
    input_data.negative_tweets = preprocess_single_tweet(input_data.negative_tweets)
    input_data.test_tweets = preprocess_single_tweet(input_data.test_tweets)

    data_pos= {"text": input_data.positive_tweets, "sentiment": np.repeat(1, len(input_data.positive_tweets))}
    data_neg= {"text": input_data.negative_tweets, "sentiment": np.repeat(0, len(input_data.negative_tweets))}
    data_test = {"text": input_data.test_tweets, "sentiment": np.repeat(-1, len(input_data.test_tweets))}
    df_pos = pd.DataFrame(data_pos)
    df_neg = pd.DataFrame(data_neg)
    test_df = pd.DataFrame(data_test)

    train_df = pd.DataFrame(columns=("text", "sentiment"))
    train_df = train_df.append(df_pos)
    train_df = train_df.append(df_neg)
    train_df = train_df.reset_index(drop = True)
    train_df["text"] = train_df.apply(lambda x: x['text'].replace("<user>", ""),axis=1)
    train_df["text"] = train_df.apply(lambda x: x['text'].replace("<url>", ""),axis=1)

    # build tokenizer 
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    maxlen = input_data.max_length
    tokenizer = Tokenizer(token_dict)
    X_data = [tokenizer.encode(x, max_len = maxlen)[0] for x in train_df["text"]]
    X_data = np.asarray(X_data).astype(np.float32)
    y_data = train_df["sentiment"].astype(np.float32)

    X_test = [tokenizer.encode(x, max_len = maxlen)[0] for x in test_df["text"]]
    X_test = np.asarray(X_test).astype(np.float32)

    reconstructed_model = load_model(load_path, custom_objects=get_custom_objects())
    print("Model Loaded, starting predictions")
    predictions = reconstructed_model.predict(X_test, verbose = 2)
    return predictions
    