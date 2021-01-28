import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import EarlyStopping 
import tensorflow as tf
import csv
import argparse
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData

# Run Model and Returns model predictions
def run_model(args, input_data):
    voc_size = input_data.voc_size
    embedding_dim = input_data.embedding_dimension
    max_len = input_data.max_length

    model = Sequential()
    #we pass in tweets as a bag-of-words vector
    model.add(Embedding(voc_size, embedding_dim, input_length = max_len))
    #we get out an embedding of each word in 256 dimensions
    model.add(tf.keras.layers.Reshape((max_len, embedding_dim, 1)))
    input_shape = (None, max_len, embedding_dim, 1)
    model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=8, strides=(2,2), padding='valid', data_format="channels_last", input_shape = input_shape,
    dilation_rate=1, activation="relu", use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=tf.keras.regularizers.l2(
    l=0.01
    ), activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None)
    )

    model.add(tf.keras.layers.MaxPool2D(
    pool_size=8, strides=(2,2), padding='valid'
    ))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"]) 
    EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

    model.fit(input_data.training_data, input_data.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_data.validation_data, input_data.validation_label), verbose = 2)

    return model, model.predict(np.array(input_data.test_data))