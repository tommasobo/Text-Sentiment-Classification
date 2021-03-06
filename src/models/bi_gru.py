import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout, Bidirectional, BatchNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras_multi_head import MultiHead
import csv
from pathlib import Path
import argparse
import os, sys
import pathlib
#from attention import attention_3d_block

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData

# Run Model and Returns model predictions
def run_model(args, input_data):

    filename = "{}_epoch{}_valacc:{}.hdf5".format(args.dataset_size, "{epoch:02d}", "{val_accuracy:.2f}",)
    filepath = Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type) / filename)
    Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type)).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(str(filepath), monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', sav_freq=1, save_weights_only=False)

    # reference paper: Sentiment Analysis of Text Based on Bidirectional LSTM With Multi-Head Attention
    emb = Embedding(input_data.voc_size, 256, input_length = 60)
    model = Sequential()
    model.add(Input(shape=(60, )))
    model.add(emb)
    model.add(Dropout(.25))
    #model.add(LSTM(units=256))
    model.add(Bidirectional(GRU(units=256, return_sequences=True, dropout = 0.2)))
    model.add(Bidirectional(GRU(units=128, return_sequences=True, dropout = 0.2)))
    model.add(Dropout(.25))
    model.add(Conv1D(filters = 128, kernel_size = 5, activation='relu'))
    model.add(Dropout(.25))
    model.add(MaxPooling1D(pool_size = 5))
    model.add(Dropout(.25))
    model.add(Conv1D(filters = 64, kernel_size = 5, activation='relu'))
    model.add(Dropout(.25))
    model.add(MaxPooling1D(pool_size = 5))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(.25))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if (args.no_save_best_epochs):
        model.fit(input_data.training_data, input_data.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_data.validation_data, input_data.validation_label), verbose = 1)
    else:
        model.fit(input_data.training_data, input_data.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_data.validation_data, input_data.validation_label), verbose = 1, callbacks=[checkpoint])

    return model, model.predict(np.array(input_data.test_data))