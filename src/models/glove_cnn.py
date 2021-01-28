# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:46:19 2020

@author: Domi
"""

import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
from pathlib import Path
import argparse
import os, sys
import pathlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData
import tensorflow as tf

'''
Basic RNN-Model Using Pretrained Glove
'''

def run_model(args, input_data):
    
    # the following embedding part credits to Dominik 
    filename = "{}_epoch{}_valacc:{}.hdf5".format(args.dataset_size, "{epoch:02d}", "{val_accuracy:.2f}",)
    filepath = Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type) / filename)
    Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type)).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(str(filepath), monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', sav_freq=1, save_weights_only=False)


    # reference:  https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    model = Sequential()
    model.add(Embedding(input_data.voc_size, input_data.embedding_dimension, input_length = input_data.max_length, weights = [input_data.embedding_matrix], trainable=True))
    model.add(Conv1D(filters = 128, kernel_size = 5, activation='relu'))
    model.add(MaxPooling1D(pool_size = 5))

    model.add(Conv1D(filters = 128, kernel_size = 5, activation='relu'))
    model.add(MaxPooling1D(pool_size = 5))
    '''
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    '''
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    if (args.no_save_best_epochs):
        model.fit(input_data.training_data, input_data.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_data.validation_data, input_data.validation_label), verbose = 1)
    else:
        model.fit(input_data.training_data, input_data.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_data.validation_data, input_data.validation_label), verbose = 1, callbacks=[checkpoint])

    return model, model.predict(np.array(input_data.test_data))