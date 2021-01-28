import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import csv
from pathlib import Path
import nltk
import argparse
from sklearn.model_selection import train_test_split
import os, sys
import pathlib
import keras
import pickle
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Embedding
import random
import keras.layers as layers
import keras.regularizers as reg
from keras.optimizers import Adam
from keras.callbacks import Callback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
from models.elmo_embedding import *
import preprocessing
from preprocessing import InputData, preprocess_single_tweet


# Part of the Elmo Embedding adapted from this guide https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
# Run Model and Returns model predictions
def run_model(args, input_obj):

    # Saving After each Epoch
    filename = "batch_size{}_data{}_epoch{}_.hdf5".format(args.batch_size, args.dataset_size, "{epoch:02d}",)
    filepath = Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type) / filename)
    Path(constants.SAVED_MODELS_FOLDER_FROM_MODELS / str(args.model_type)).mkdir(parents=True, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(str(filepath), monitor='acc', verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=False)

    # Defining Model 
    # Not working because of some masking issue, needs to be checked if possible
    inputs = layers.Input(shape=(constants.MAX_WORDS_IN_SENTENCE, ), dtype="string")
    #mask = layers.Masking().compute_mask(inputs) # <= Compute the mask
    previous_out = ElmoEmbeddingAbstract.build(elmo_type='elmo', batch=args.batch_size, trainable=True)(inputs)
    previous_out = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(previous_out)
    previous_out = layers.Conv1D(filters=64,
                    kernel_size=3, 
                    padding="valid", 
                    kernel_initializer="he_uniform")(previous_out)    
    previous_out = layers.Dropout(0.4)(previous_out)
    final_out = layers.Dense(1, activation='sigmoid')(previous_out)

    model = Model(inputs=inputs, outputs=final_out)
    print(model.summary())
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    if (args.no_save_best_epochs):
        model.fit(input_obj.training_data, input_obj.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_obj.validation_data, input_obj.validation_label), verbose = 2)
    else:
        model.fit(input_obj.training_data, input_obj.training_label, batch_size = args.batch_size, epochs = args.number_epochs, validation_data = (input_obj.validation_data, input_obj.validation_label), verbose = 2, callbacks=[checkpoint])

    return model, model.predict(np.array(input_obj.test_data), batch_size = args.batch_size)