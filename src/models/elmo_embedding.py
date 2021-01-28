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
from keras.optimizers import Adam
from keras.callbacks import Callback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
import preprocessing
from preprocessing import InputData, preprocess_single_tweet

# Part of the Elmo Embedding adapted from this guide https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
class ElmoEmbeddingAbstract(Embedding):

    @staticmethod
    def build(elmo_type, batch, trainable):
        return ElmoEmbedding(elmo_type = elmo_type, batch_size = batch, trainable = trainable)

class ElmoEmbedding(Layer):

    def __init__(self, trainable = True, batch_size = 32,  elmo_type = 'default', **kwargs):
        self.batch_size = batch_size
        self.trainable = trainable
        self.dimensions = 1024
        self.type = elmo_type
        super(ElmoEmbedding, self).__init__(**kwargs)


    def build(self, input_shape):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable = self.trainable, name = "{}_module".format(self.name))
        self.trainable_weights += tf.trainable_variables(scope = "^{}_module/.*".format(self.name))
        super(ElmoEmbedding, self).build(input_shape)


    def call(self, x, mask = None):
        return self.elmo(
            inputs={
                'tokens': K.cast(x, tf.string),
                'sequence_len': tf.constant([constants.MAX_WORDS_IN_SENTENCE] * self.batch_size)
            }, as_dict = True, signature = 'tokens', )[self.type]


    def compute_mask(self, inputs, mask = None):
        return K.not_equal(inputs, '--PAD--')


    def compute_output_shape(self, input_shape):
        return (None, constants.MAX_WORDS_IN_SENTENCE, self.dimensions)