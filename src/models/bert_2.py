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

import re, os
import codecs

from tensorflow.python import keras
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData, preprocess_single_tweet
from tensorflow.python.ops.math_ops import erf, sqrt


maxlen = 50
config_path = '../data/input/uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '../data/input/uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '../data/input/uncased_L-24_H-1024_A-16/vocab.txt'

def run_model(args, input_data):

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

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.015, shuffle = True, random_state = 42 if args.fix_shuffle else None)    
    D_train = tensorflow.data.Dataset.from_tensor_slices((X_train, y_train))
    D_train = D_train.shuffle(y_train.shape[0])
    D_train = D_train.batch(BATCH_SIZE)
    D_train = D_train.repeat()

    D_val = tensorflow.data.Dataset.from_tensor_slices((X_val, y_val))
    D_val = D_val.batch(BATCH_SIZE)
    # get pretrained model 
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, trainable=True, seq_len=maxlen)
    (indices, zeros) = bert_model.inputs[:2]
    
    dense = bert_model.get_layer('NSP-Dense').output
    bert_model = keras.models.Model(
        inputs=(indices, zeros),
        outputs=dense
    )
    # build our model 
    inputs = indices
    bert = bert_model([inputs, tensorflow.zeros_like(inputs)])
    bert = keras.layers.Dropout(0.3)(bert)
    bert = keras.layers.Dense(32, activation='relu')(bert)
    bert = keras.layers.Dropout(0.3)(bert)
    bert = keras.layers.Dense(8, activation='relu')(bert)
    bert = keras.layers.Dropout(0.3)(bert)
    bert = keras.layers.Dense(1, activation='sigmoid')(bert)
    model = keras.models.Model(inputs=indices, outputs=bert)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.000005), # learning rate that is small enough
        metrics=['accuracy']
    )

    # Saving model with best validation acc
    filename = "batch_size{}_data{}_steps{}_epoch{}.hdf5".format(args.batch_size, args.dataset_size, STEPS, "{epoch:02d}")
    filepath = Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type) / filename)
    Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type)).mkdir(parents=True, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(str(filepath), monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)

    if (args.no_save_best_epochs):
        model.fit( D_train, validation_data=D_val, epochs=EPOCHS, steps_per_epoch=STEPS,  verbose = 1)
    else:
        model.fit( D_train, validation_data=D_val, epochs=EPOCHS, steps_per_epoch=STEPS,  verbose = 1, callbacks=[checkpoint])

    return model, model.predict(X_test)