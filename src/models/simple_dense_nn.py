import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import EarlyStopping 
import csv
import argparse
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData

# Run Model and Returns model predictions
def run_model(args, input_data):

    model = Sequential()
    model.add(Embedding(input_data.voc_size, input_data.embedding_dimension, input_length = input_data.max_length))
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.38))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.38))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

    model.fit(input_data.training_data, input_data.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_data.validation_data, input_data.validation_label), verbose = 1)

    return model, model.predict(np.array(input_data.test_data))