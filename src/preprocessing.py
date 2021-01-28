# Preprocessing common to ALL models should go here. Preprocessing specific to a model should go inside /models/
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import tensorflow as tf
import re, os
import pickle
import constants

# Enable all pre-processing
REMOVE_EMOCTIONS = True 
CONVERT_NUMBERS = True
CONVERT_KISSES = True
MARK_ELONGATED = True

class InputData:
    def __init__(self):
        self.training_data = None
        self.validation_data = None
        self.training_label = None
        self.validation_label = None
        self.test_data = None
        self.voc_size = None
        self.embedding_dimension = None
        self.max_length = None
        self.positive_tweets = None
        self.negative_tweets = None
        self.test_tweets = None
        self.embedding_matrix = None


'''
    Returns a list of preprocessed tweets. In particular many preprocessing steps are taken depending on the boolean flags set.
    Args:
        list_tweets: list of tweets to be used.
'''
def preprocess_single_tweet(list_tweets):
    # Do stuff to single tweet like handle emojii
    # Definitions Taken from Wikipedia (only main ones)
    emoctions_map = {"happy": [":‑)", ":-]", ":-3", ":->", "8-)", ":-}", ":o)", ":c)", ":^)", "=)",
                  ":)", ":]", ":3", ":>", "8)", ":}", ":=)", "\o/", ":-))"],
            "laughing": [":‑d", "8‑d", "x‑d", "x‑d", "=d", "=3", "b^d", "xd", "rofl", "roflmao"],
            "sad": [":‑(", ":‑c", ":‑<", ":‑[", ":-||", ">:[", ":{", ":@", ":(", ":c", ":<", ":[", ""],
            "crying": [":'‑(", ":'("],
            "disgust": ["D‑':", "D:<", "D:", "D8", "D;", "D=", "DX"],
            "surprise": [":‑O", ":‑o", "D:", ":-0", "8‑0", ":o", "o_o", ":O"],
            "kissing": [":-*", ":x", ":*"],
            "winking": [";‑)", "*-)", ";)", ";d"],
            "annoyed": [":‑/", ":/", ":‑."],
            "angel": ["o:‑)", "o:)"],
            "cheeky" : [":p", "xp"],
            "love": ["<3"]
            }

    if (REMOVE_EMOCTIONS):
        for index, tweet in enumerate(list_tweets):
            for key_map in emoctions_map.keys():
                for index_emoction, emoction in enumerate(emoctions_map[key_map]):
                    list_tweets[index] = tweet.replace(emoction, " " + key_map + " ")
            
            # adapted from: https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
            if (CONVERT_NUMBERS):
                list_tweets[index] = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", tweet) #confused about the ':' inclusion..
            if (CONVERT_KISSES):
                list_tweets[index] = re.sub(r"\bx{2,10}\b", r" kisses ", tweet) # people often use x as a multiplier as well, so at least 2
                list_tweets[index] = re.sub(r"\b(xo|ox)+\b", r" hugs and kisses ", tweet)
            if(MARK_ELONGATED):
                list_tweets[index] = re.sub(r"\b(\S*?)(.)\2{3,}\b", r"\1\2 <elong>", tweet)
                    
    return list_tweets


'''
    Saves to file a python object
    Args:
        content: object to save
        name: name of the file to save
'''
def save_to_file(content, name):
    # Save preprocessed data to File
    with open(name, 'wb') as f:
        pickle.dump(content, f)


'''
    Read a python object from file
    Args:
        name: name of the file to load
'''
def read_from_file(name):
    with open (name, 'rb') as f:
        input_dat = pickle.load(f)
    return input_dat


'''
    Preprocess test data specifically for ELMo
    Args:
        name: Input data to be preprocessed
'''
def preprocess_test_elmo(input_data):
    # Prepare Raw
    test_tweets_raw = input_data.test_tweets

    # Pre Process
    test_tweets_raw = preprocess_single_tweet(test_tweets_raw)

    # Max words per tweet
    addition = lambda l: l + ['' for i in range(constants.MAX_WORDS_IN_SENTENCE - len(l))] 
    test_tweets = [addition(tweet.split()) for tweet in test_tweets_raw]
    test_tweets = pad_sequences(dtype=object, sequences=test_tweets, maxlen=constants.MAX_WORDS_IN_SENTENCE, padding='post')

    return test_tweets


'''
    Preprocess training data specifically for ELMo
    Args:
        args: command line arguments
        name: Input data to be preprocessed
'''
def preprocess_data_elmo(args, input_data):

    print("Preprocessing Data for ELMo")
    # Prepare Raw
    all_tweets_raw = input_data.positive_tweets + input_data.negative_tweets
    test_tweets_raw = input_data.test_tweets

    # Pre Process
    all_tweets_raw = preprocess_single_tweet(all_tweets_raw)
    test_tweets_raw = preprocess_single_tweet(test_tweets_raw)

    # Generate Labels 
    all_labels = np.array([1] * len(input_data.positive_tweets) + [0] * len(input_data.negative_tweets))

    # Allow a maximum of different words
    tokenizer = Tokenizer(num_words=constants.MAX_WORDS_VOC)
    tokenizer.fit_on_texts(all_tweets_raw)

    # Max words per tweet
    addition = lambda l: l + ['' for i in range(constants.MAX_WORDS_IN_SENTENCE - len(l))] 
    all_tweets_raw = [addition(tweet.split()) for tweet in all_tweets_raw]
    test_tweets = [addition(tweet.split()) for tweet in test_tweets_raw]
    all_tweets_raw = pad_sequences(dtype=object, sequences=all_tweets_raw, maxlen=constants.MAX_WORDS_IN_SENTENCE, padding='post')
    test_tweets = pad_sequences(dtype=object, sequences=test_tweets, maxlen=constants.MAX_WORDS_IN_SENTENCE, padding='post')

    # Convert from list to numpy array
    all_tweets_raw = np.array(all_tweets_raw)

    # Divide Training / Validation Split
    input_obj = InputData()
    input_obj.test_data = test_tweets
    input_obj.training_data, input_obj.validation_data, input_obj.training_label, input_obj.validation_label = train_test_split(
        all_tweets_raw, all_labels, test_size = 1 - constants.TRAINING_VALIDATION_SPLIT, shuffle = True, random_state = 42 if args.fix_shuffle else None)

    # Stupid trick to make it divisible by batch size, otherwise it breaks the architecture. Known bug of this version
    # but it's the last compatible with ELMO. Anyway it only removes a handful of examples and add some dummy data to the test which is later removed.
    rem_training = len(input_obj.training_data) % args.batch_size
    rem_validation = len(input_obj.validation_data) % args.batch_size
    
    input_obj.training_data = input_obj.training_data[:len(input_obj.training_data) - rem_training]
    input_obj.validation_data = input_obj.validation_data[:len(input_obj.validation_data) - rem_validation]

    input_obj.training_label = input_obj.training_label[:len(input_obj.training_label) - rem_training]
    input_obj.validation_label = input_obj.validation_label[:len(input_obj.validation_label) - rem_validation]

    if (len(input_obj.test_data) % args.batch_size != 0):
        rem_test = len(input_obj.test_data) % args.batch_size
        last_ele = input_obj.test_data[len(input_obj.test_data) - 1]
        for index in range ((args.batch_size - rem_test)):
            input_obj.test_data = np.vstack((input_obj.test_data, last_ele))

    return input_obj


'''
    Checks if preprocessed files have already been created. If yes it returns True, otherwise False
    Args:
        args: command line arguments
'''
def check_if_cached(args):
    # Check if we have already preprocessed the data in the past
    
    path_pre = constants.PRE_PROCESSING_FOLDER_FULL
    
    if (args.dataset_size == constants.SMALL):
        path_pre = constants.PRE_PROCESSING_FOLDER_SMALL
    
    # If we are using pretrained glove, then we need the embedding matrix as well
    if (args.model_type == constants.GLOVE_RNN or args.model_type == constants.GLOVE_CNN  ):
        constants.LIST_FILES_CHECK.append("embedding_matrix")
        path_pre = constants.PRE_PROCESSING_FOLDER_GLOVE_FULL
        if (args.dataset_size == constants.SMALL):
            path_pre = constants.PRE_PROCESSING_FOLDER_GLOVE_SMALL
            
            
    for file_name in constants.LIST_FILES_CHECK:
        my_file = Path(path_pre / file_name)
        if not my_file.is_file():
            return False

    return True
    

'''
    Save preprocessed data in a sort of cache that can be utilized during a later run.
    Args:
        args: command line arguments
'''
def build_cached_version(args):
    input_obj = InputData()
    
    path_pre = constants.PRE_PROCESSING_FOLDER_FULL
    if (args.dataset_size == constants.SMALL):
        path_pre = constants.PRE_PROCESSING_FOLDER_SMALL
        
    if (args.model_type == constants.GLOVE_RNN or args.model_type == constants.GLOVE_CNN):
        path_pre = constants.PRE_PROCESSING_FOLDER_GLOVE_FULL
        if (args.dataset_size == constants.SMALL):
            path_pre = constants.PRE_PROCESSING_FOLDER_GLOVE_SMALL
        
        input_obj.embedding_matrix = read_from_file(path_pre / constants.LIST_FILES_CHECK[11])
    

    input_obj.training_label = read_from_file(path_pre / constants.LIST_FILES_CHECK[0])
    input_obj.validation_label = read_from_file(path_pre / constants.LIST_FILES_CHECK[1])
    input_obj.training_data = read_from_file(path_pre / constants.LIST_FILES_CHECK[2])
    input_obj.validation_data = read_from_file(path_pre / constants.LIST_FILES_CHECK[3])
    input_obj.test_data = read_from_file(path_pre / constants.LIST_FILES_CHECK[4])
    input_obj.voc_size = read_from_file(path_pre / constants.LIST_FILES_CHECK[5])
    input_obj.embedding_dimension = read_from_file(path_pre / constants.LIST_FILES_CHECK[6])
    input_obj.max_length = read_from_file(path_pre / constants.LIST_FILES_CHECK[7])
    input_obj.positive_tweets = read_from_file(path_pre / constants.LIST_FILES_CHECK[8])
    input_obj.negative_tweets = read_from_file(path_pre / constants.LIST_FILES_CHECK[9])
    input_obj.test_tweets = read_from_file(path_pre / constants.LIST_FILES_CHECK[10])

    return input_obj


'''
    Run all the normal preprocessing steps for any type of model. 
    Args:
        args: command line arguments
'''
def run_preprocessing(args):

    # Check if we have cached version
    if (check_if_cached(args)):
        if (args.overwrite_cached_data is False):
            print("Preprocessed cached files detected, using them for the model run.")
            input_obj = build_cached_version(args)
            return input_obj
        else:
            print("Preprocessed cached files detected but overwriting them due to --overwrite_cached_data flag.")
    else:
        print("No preprocessed cached files detected, running full preprocessing.")

    # Check whether we are using small or full dataset
    if (args.dataset_size == constants.SMALL):
        train_neg_path = constants.TRAIN_NEG_DATA_SMALL
        train_pos_path = constants.TRAIN_POS_DATA_SMALL
    else:
        train_neg_path = constants.TRAIN_NEG_DATA_FULL
        train_pos_path = constants.TRAIN_POS_DATA_FULL

    # Load Tweets
    positive_tweets = Path(train_pos_path)
    negative_tweets = Path(train_neg_path)
    test_tweets = Path(constants.TEST_DATA)

    # Load text of tweets in a list splitted by new line (new tweet)
    positive_tweets = positive_tweets.read_text().split("\n")[:-1]
    # Dominik:: I got a UnicodeDecodeError when handling the full dataset without defining encoding
    negative_tweets = negative_tweets.read_text(encoding = "utf8").split("\n")[:-1]
    test_tweets = test_tweets.read_text().split("\n")
    test_array = list()
    for line in test_tweets:
        trim = line.find(",") + 1
        line = line[trim:]
        test_array.append(line)

    # Remove Duplicates; ettlinc: changed implementation slightly to ensure reproducibility (turning the list into a set destroys the order)
    positive_tweets =  list(dict.fromkeys(positive_tweets)) #list(set(positive_tweets))
    negative_tweets = list(dict.fromkeys(negative_tweets)) #list(set(negative_tweets))

    # Remove Empty Tweets
    positive_tweets = list(filter(None, positive_tweets))
    negative_tweets = list(filter(None, negative_tweets))
    test_array = list(filter(None, test_array))

    # Merge Into a single list
    all_tweets = positive_tweets + negative_tweets

    # Generate Labels 
    positive_labels = [1] * len(positive_tweets)
    negative_labels = [0] * len(negative_tweets)
    all_labels = positive_labels + negative_labels

    # Process Single Tweets
    all_tweets = preprocess_single_tweet(all_tweets)
    test_array = preprocess_single_tweet(test_array)

    # Divide Training / Validation Split
    input_obj = InputData()
        
    t_data, v_data, t_label, v_label = train_test_split(
        all_tweets, all_labels, test_size = 1 - constants.TRAINING_VALIDATION_SPLIT, shuffle = True, random_state = 42 if args.fix_shuffle else None)
    
    input_obj.training_data, input_obj.validation_data, input_obj.training_label, input_obj.validation_label = t_data, v_data, t_label, v_label
    # Tokenize words 
    tokenizer = Tokenizer() 
    all_text = np.concatenate([ input_obj.training_data , input_obj.validation_data, np.array(test_array)]) # Add test Array
    tokenizer.fit_on_texts(all_text)
    X_train_token = tokenizer.texts_to_sequences(input_obj.training_data)
    X_test_token = tokenizer.texts_to_sequences(input_obj.validation_data)
    test_array_token = tokenizer.texts_to_sequences(np.array(test_array))

    # Pad Sequences
    voc_size = len(tokenizer.word_index)+1 
    max_len = 60
    X_train_pad = pad_sequences(X_train_token, maxlen = max_len, padding = "post" )
    X_test_pad = pad_sequences(X_test_token, maxlen = max_len, padding = "post" )
    X_test_array_pad = pad_sequences(test_array_token, maxlen = max_len, padding = "post" )

    # Convert to NP Arrays
    training_label_nparray = np.asarray(input_obj.training_label).astype(np.float32)
    validation_label_nparray = np.asarray(input_obj.validation_label).astype(np.float32)

    # Create GLOVE dict
    # adapted from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    if args.model_type == constants.GLOVE_RNN or args.model_type == constants.GLOVE_CNN:
        print("Creating Glove dictionary")
        embedding_dim = constants.GLOVE_DIMENSIONS
        with open(constants.PRETRAINED_GLOVE_EMBEDDING, 'r', encoding="utf8") as f:
            vocab = {}
            print("Words processed:")
            for idx, line in enumerate(f):
                tokens = line.split()
                coefficients = np.asarray(tokens[1:], "float32")
                vocab[tokens[0]] = coefficients
                if idx % 100000 == 0: print(idx)
        em = np.zeros((voc_size, embedding_dim))
        for word, i in tokenizer.word_index.items():
            em_vector = vocab.get(word)
            if em_vector is not None:
                em[i] = em_vector


    # Save to Class and file
    print("Preprocessing done. Saving preprocessed data to files for future use.")
    path_pre = constants.PRE_PROCESSING_FOLDER_FULL
    if (args.dataset_size == constants.SMALL):
        path_pre = constants.PRE_PROCESSING_FOLDER_SMALL
    path = Path(path_pre)
    
    if (args.model_type == constants.GLOVE_RNN or args.model_type == constants.GLOVE_CNN):
        path_pre = constants.PRE_PROCESSING_FOLDER_GLOVE_FULL
        if (args.dataset_size == constants.SMALL):
            path_pre = constants.PRE_PROCESSING_FOLDER_GLOVE_SMALL
        path = Path(path_pre)
    
    path.mkdir(parents=True, exist_ok=True)

    input_obj.training_label = training_label_nparray
    save_to_file(training_label_nparray, path_pre / "training_label")

    input_obj.validation_label = validation_label_nparray
    save_to_file(validation_label_nparray, path_pre / "validation_label")

    input_obj.training_data = X_train_pad
    save_to_file(X_train_pad, path_pre / "training_data")

    input_obj.validation_data = X_test_pad
    save_to_file(X_test_pad, path_pre / "validation_data")

    input_obj.test_data = X_test_array_pad
    save_to_file(X_test_array_pad, path_pre / "test_data")

    input_obj.voc_size = voc_size
    save_to_file(voc_size, path_pre / "voc_size")


    if (args.model_type == constants.GLOVE_RNN or args.model_type == constants.GLOVE_CNN):
        input_obj.embedding_dimension = constants.GLOVE_DIMENSIONS
        save_to_file(constants.GLOVE_DIMENSIONS, path_pre / "embedding_dimension")
    else:
        input_obj.embedding_dimension = constants.EMBEDDING_DIMENSION
        save_to_file(constants.EMBEDDING_DIMENSION, path_pre / "embedding_dimension")
    
    input_obj.max_length = max_len
    save_to_file(max_len, path_pre / "max_length")

    input_obj.positive_tweets = positive_tweets
    save_to_file(positive_tweets, path_pre / "positive_tweets")

    input_obj.negative_tweets = negative_tweets
    save_to_file(negative_tweets, path_pre / "negative_tweets")

    input_obj.test_tweets = test_array
    save_to_file(test_array, path_pre / "test_tweets")
    

    if (args.model_type == constants.GLOVE_RNN or args.model_type == constants.GLOVE_CNN):
        input_obj.embedding_matrix = em
        save_to_file(em, path_pre / "embedding_matrix")
    

    # Return Data
    return input_obj