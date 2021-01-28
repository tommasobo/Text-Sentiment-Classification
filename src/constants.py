from pathlib import Path

# FOLDERS 
DATA_FOLDER = Path("../data/")
INPUT_DATA_FOLDER = DATA_FOLDER / "input/"
OUTPUT_DATA_FOLDER = DATA_FOLDER / "output/"
SAVED_MODELS_FOLDER = DATA_FOLDER / "saved_models/"
SAVED_MODELS_FOLDER_FROM_MODELS = Path("../data/saved_models/")
PRE_PROCESSING_FOLDER_FULL = DATA_FOLDER / "pre_processing_full_dataset/"
PRE_PROCESSING_FOLDER_SMALL = DATA_FOLDER / "pre_processing_small_dataset/"
PRE_PROCESSING_FOLDER_GLOVE_SMALL = DATA_FOLDER / "pre_processing_small_glove/"
PRE_PROCESSING_FOLDER_GLOVE_FULL = DATA_FOLDER / "pre_processing_full_glove/"
ENSEMBLE_FOLDER = DATA_FOLDER / "ensemble/"
ENSEMBLE_FOLDER_SUBMISSION = ENSEMBLE_FOLDER / "submission/"
SCRATCH_FOLDER = Path("/cluster/scratch/")
SAVED_MODELS_FOLDER_CLUSTER = "saved_models/"
VALIDATE_ENSEMBLE = DATA_FOLDER / "validate_ensemble/"

# INPUT DATA
TRAIN_POS_DATA_SMALL = INPUT_DATA_FOLDER / "train_pos.txt"
TRAIN_NEG_DATA_SMALL = INPUT_DATA_FOLDER / "train_neg.txt"
TRAIN_POS_DATA_FULL = INPUT_DATA_FOLDER / "train_pos_full.txt"
TRAIN_NEG_DATA_FULL = INPUT_DATA_FOLDER / "train_neg_full.txt"
TEST_DATA = INPUT_DATA_FOLDER / "test_data.txt"
GLOVE_VOCAB = INPUT_DATA_FOLDER / "glove_vocab.pkl"
GLOVE_COOC = INPUT_DATA_FOLDER / "glove_cooc.pkl"
GLOVE_VOCAB_TXT = INPUT_DATA_FOLDER / "glove_vocab.txt"
GLOVE_EMBEDDING = INPUT_DATA_FOLDER / "glove_embedding_200d.npy"
GLOVE_EMBEDDING_50 = INPUT_DATA_FOLDER / "glove_embedding_50d.npy"
PRETRAINED_GLOVE_EMBEDDING = INPUT_DATA_FOLDER / "glove.twitter.27B.200d.txt"
PRETRAINED_GLOVE_EMBEDDING_50 = INPUT_DATA_FOLDER / "glove.twitter.27B.50d.txt"

# OUTPUT DATA
SUBMISSION_PATH = OUTPUT_DATA_FOLDER 

# RANDOM SEED VALUE
RANDOM_SEED = 42

# NN SPECIFIC
TRAINING_VALIDATION_SPLIT = 0.90
EMBEDDING_DIMENSION = 256 
MAX_WORDS_IN_SENTENCE = 55
MAX_WORDS_VOC = 30000
NUM_TEST_SAMPLES = 10000

# GLOVE SPECIFIC
GLOVE_DIMENSIONS = 200
GLOVE_PRETRAINED_TWEETS = "pretrained"

'''
    From now on we give a list of Models which have been used for the final papers. The others models are still available later
'''

# BASELINES
BASIC_NN = "basic_nn"
BASELINE_GLOVE = "baseline_glove"
BASELINE_GLOVE_50 = "baseline_glove_50"
BASELINE_WORD2VEC = "baseline_word2vec"

# NORMAL KERAS MODELS
BI_LSTM = "bi_lstm"
SIMPLE_CNN = "simple_cnn"
CNN_RNN = "cnn_rnn"

# ELMo MODELS
ELMO_LSTM = "elmo_lstm"
ELMO_GRU = "elmo_bigru"
ELMO_BILSTM_3 = "elmo_bilstm_3"
ELMO_BILSTM_MULTI_ATTENTION = "elmo_bilstm_multi_attention"

#BERT MODELS
# TBD

'''
    Rest of the models.
'''
# MODEL OPTIONS 
SIMPLE_DENSE = "simple_dense_nn"
SIMPLE_RNN = "simple_rnn"
GLOVE_RNN = "glove_rnn"
GLOVE_CNN = "glove_cnn"
BI_GRU = "bi_gru"
SMALL = "small"
FULL = "full"
BI_LSTM_TESTING = "bi_lstm_testing"

# ELMO MODELS
ELMO_LSTM = "elmo_lstm"
ELMO_LSTM_1 = "elmo_lstm_1"
ELMO_LSTM_1_SPATIAL = "elmo_lstm_1_spatial"
ELMO_BILSTM = "elmo_bilstm"
ELMO_CNN_RNN = "elmo_cnn_rnn"
ELMO_BILSTM_1 = "elmo_bilstm_1"
ELMO_BILSTM_4 = "elmo_bilstm_4"
ELMO_BILSTM_ATTENTION = "elmo_bilstm_attention"
ELMO_SIMPLE = "elmo_simple"

# BERT MODELS
BERT = "bert"
BERT_2 = "bert_2"
BERT_3 = "bert_3"
BERT_4 = "bert_4"
BERT_5 = "bert_5"
BERT_TRAIN_MORE = "bert_train_more"
BERT_6 = "bert_6"

# OTHER
LIST_FILES_CHECK = ["training_label", "validation_label", "training_data", "validation_data", "test_data", "voc_size", 
                               "embedding_dimension", "max_length", "positive_tweets", "negative_tweets", "test_tweets"]

