#random seed imports & setting (done first since previous imports can apparently influence the randomness already)
import os
os.environ['TF_KERAS'] = '1'
import random
import numpy as np
import tensorflow as tf

#set seed manually here (since we do not also want to import constants beforehand), and set to same value in constants (use that in models)
seed_val = 42
os.environ['PYTHONHASHSEED']=str(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

# Check Version to use correct seed function and import compatible only with one version
if tf.__version__ == '1.13.1':
    tf.set_random_seed(seed_val)
else:
    tf.random.set_seed(seed_val)
    from models.bert import run_model as run_model_bert
    from models.bert_2 import run_model as run_model_bert_2
    from models.bert_3 import run_model as run_model_bert_3
    from models.bert_4 import run_model as run_model_bert_4
    from models.bert_5 import run_model as run_model_bert_5
    from models.bert_6 import run_model as run_model_bert_6
    from models.predict_bert import predict_bert
    from models.load_bert import run_model as load_bert_training

# Other imports
import csv
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model
from keras.models import load_model as load_model_keras
from sklearn.model_selection import train_test_split
import pickle

# Models Import (Baselines + Keras)
from models.basic_nn import run_model as run_model_basic_nn
from models.simple_dense_nn import run_model as run_model_simple_dense_nn
from models.simple_rnn import run_model as run_model_simple_rnn
from models.baseline_word2vec import run_model as run_model_baseline_word2vec
from models.simple_cnn import run_model as run_model_simple_cnn
from models.baseline_glove import run_model as run_model_baseline_glove
from models.baseline_glove_50d import run_model as run_model_baseline_glove_50
from models.glove_rnn import run_model as run_model_glove_rnn
from models.glove_cnn import run_model as run_model_glove_cnn
from models.bi_lstm import run_model as run_model_bi_lstm
from models.bi_gru import run_model as run_model_bi_gru
from models.cnn_rnn import run_model as run_model_cnn_rnn
from models.glove_cnn import run_model as run_model_glove_cnn
from models.bi_lstm_testing import run_model as run_model_bilstm_test

# ELMO Models
from models.elmo_lstm import ElmoEmbedding, ElmoEmbeddingAbstract
from models.elmo_lstm import run_model as run_model_elmo_lstm
from models.elmo_lstm_1 import run_model as run_model_elmo_lstm_1
from models.elmo_lstm_1_spatial import run_model as run_model_elmo_lstm_1_spatial
from models.elmo_bi_lstm import run_model as run_model_elmo_bi_lstm
from models.elmo_cnn_rnn import run_model as run_model_elmo_cnn_rnn
from models.elmo_bi_lstm_1 import run_model as run_model_elmo_bi_lstm_1
from models.elmo_gru import run_model as run_model_elmo_gru
from models.elmo_bi_lstm_3 import run_model as run_model_elmo_bi_lstm_3
from models.elmo_bi_lstm_4 import run_model as run_model_elmo_bi_lstm_4
from models.elmo_bilstm_attention import run_model as run_model_elmo_bi_lstm_attention
from models.elmo_bi_lstm_multi_attention import run_model as run_model_elmo_bi_lstm_multi_attention
from models.elmo_simple import run_model as run_model_elmo_simple

# Other custom imports
import constants
from ensemble import run_ensemble
from preprocessing import InputData, run_preprocessing, preprocess_test_elmo, preprocess_data_elmo


'''
    Matches the selected (via CLI) machine learning model to use for the training or/and predictions.
    Args:
        args: command line arguments
        model_data: input data to be passed to the model
'''
def match_model(args, model_data):
    print("Matching Model and Launching it\n")

    if (args.model_type == constants.BASIC_NN):
        return run_model_basic_nn(args, model_data)
    elif (args.model_type == constants.SIMPLE_DENSE):
        return run_model_simple_dense_nn(args, model_data)
    elif (args.model_type == constants.SIMPLE_RNN):
        return run_model_simple_rnn(args, model_data)
    elif (args.model_type == constants.BASELINE_WORD2VEC):
        return run_model_baseline_word2vec(args, model_data)
    elif (args.model_type == constants.SIMPLE_CNN):
        return run_model_simple_cnn(args, model_data)
    elif (args.model_type == constants.BASELINE_GLOVE):
        return run_model_baseline_glove(args, model_data)
    elif (args.model_type == constants.BASELINE_GLOVE_50):
        return run_model_baseline_glove_50(args, model_data)
    elif (args.model_type == constants.BERT):
        return run_model_bert(args, model_data)
    elif (args.model_type == constants.GLOVE_RNN):
        return run_model_glove_rnn(args, model_data)
    elif (args.model_type == constants.GLOVE_CNN):
        return run_model_glove_cnn(args, model_data)
    elif (args.model_type == constants.BI_LSTM):
        return run_model_bi_lstm(args, model_data)
    elif (args.model_type == constants.BI_GRU):
        return run_model_bi_gru(args, model_data)
    elif (args.model_type == constants.CNN_RNN):
        return run_model_cnn_rnn(args, model_data)
    elif (args.model_type == constants.BI_LSTM_TESTING):
        return run_model_bilstm_test(args, model_data)
    elif (args.model_type == constants.GLOVE_CNN):
        return run_model_glove_cnn(args, model_data)
    elif (args.model_type == constants.ELMO_LSTM):
        return run_model_elmo_lstm(args, model_data)
    elif (args.model_type == constants.ELMO_LSTM_1):
        return run_model_elmo_lstm_1(args, model_data)
    elif (args.model_type == constants.ELMO_LSTM_1_SPATIAL):
        return run_model_elmo_lstm_1_spatial(args, model_data)
    elif (args.model_type == constants.ELMO_BILSTM):
        return run_model_elmo_bi_lstm(args, model_data)
    elif (args.model_type == constants.ELMO_CNN_RNN):
        return run_model_elmo_cnn_rnn(args, model_data)
    elif (args.model_type == constants.ELMO_BILSTM_1):
        return run_model_elmo_bi_lstm_1(args, model_data)
    elif (args.model_type == constants.ELMO_GRU):
        return run_model_elmo_gru(args, model_data)
    elif (args.model_type == constants.ELMO_BILSTM_3):
        return run_model_elmo_bi_lstm_3(args, model_data)
    elif (args.model_type == constants.ELMO_BILSTM_4):
        return run_model_elmo_bi_lstm_4(args, model_data)
    elif (args.model_type == constants.ELMO_BILSTM_ATTENTION):
        return run_model_elmo_bi_lstm_attention(args, model_data)
    elif (args.model_type == constants.ELMO_BILSTM_MULTI_ATTENTION):
        return run_model_elmo_bi_lstm_multi_attention(args, model_data)
    elif (args.model_type == constants.ELMO_SIMPLE):
        return run_model_elmo_simple(args, model_data)
    elif (args.model_type == constants.BERT_2):
        return run_model_bert_2(args, model_data)
    elif (args.model_type == constants.BERT_3):
        return run_model_bert_3(args, model_data)
    elif (args.model_type == constants.BERT_4):
        return run_model_bert_4(args, model_data)
    elif (args.model_type == constants.BERT_5):
        return run_model_bert_5(args, model_data)
    elif (args.model_type == constants.BERT_6):
        return run_model_bert_6(args, model_data)
    elif (args.model_type == constants.BERT_TRAIN_MORE):
        return load_bert_training(args, model_data, Path(str(args.load_name))) 
    else:
        print("Model Name not found, check options in constants.py or READme")
        exit()


'''
    Writes to file the predictions made. Also builds the right header and structure.
    Args:
        prediction: list of predictions.
        filename: name to use to save the file.
'''
def write_to_file(prediction, filename):
    with open(filename, 'w', newline='') as file:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        i = 1
        for elem in prediction:
            if (elem >= 0.5):
                writer.writerow({'Id': i, 'Prediction': "1"})
            else:
                writer.writerow({'Id': i, 'Prediction': "-1"})
            i += 1


'''
    Whether we want to reshuffle training and validationd data
    Args:
        in_data: data to be reshuffled
        fixed: whether we want to use a fixed seed or not
'''
def reshuffle(in_data, fixed):
    data = np.concatenate((in_data.training_data, in_data.validation_data))
    labels = np.concatenate((in_data.training_label, in_data.validation_label))

    in_data.training_data, in_data.validation_data, in_data.training_label, in_data.validation_label = train_test_split(
        data, labels, test_size = 1 - constants.TRAINING_VALIDATION_SPLIT, shuffle = True, random_state = 42 if fixed else None)

    return in_data


'''
    Save two submission files, one to be saved as latest and the other in a folder specific to the model type
    Args:
        prediction: list of predictions to be saved
        args: command line arguments
'''
def save_predictions(prediction, args):
    if (len(prediction) > constants.NUM_TEST_SAMPLES):
        prediction = prediction[0:constants.NUM_TEST_SAMPLES]
    constants.SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    filename = "submission-"+ str(args.model_type) + "-" +str(args.dataset_size) + "-"+ str(args.number_epochs) + "-"+ str(args.batch_size) + ".csv"
    write_to_file(prediction, constants.SUBMISSION_PATH / "latest_submission.csv")
    path = Path(constants.OUTPUT_DATA_FOLDER / str(args.model_type))
    path.mkdir(parents=True, exist_ok=True)
    write_to_file(prediction, constants.OUTPUT_DATA_FOLDER / str(args.model_type) / filename)


def main():
    # Disable not so useful debugging messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf.get_logger().setLevel('INFO')

    # Parse Arguments and Print State
    parser = argparse.ArgumentParser(description='Run Sentimental Analysis')
    parser.add_argument("model_type", type=str, default="basic_nn", help="Name of the model to run")
    parser.add_argument("--dataset_size", type=str, default="full", help="Which dataset to use (options are small or full)")
    parser.add_argument("--number_epochs", type=int, default=3, help="Number of epochs. Default is 3")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training. Default is 128")
    parser.add_argument("--save_model", action='store_true', help="Use this option to save the NN model")
    parser.add_argument("--glove_model", type=str, default="pretrained", help="Only relevant for glove model. Whether we want to used pretrained glove or train it ourselves. (options are learned or pretrained. Default is pretrained")
    parser.add_argument("--no_save_best_epochs", action='store_true', help="Use this option to disable saving the full model after each epoch where there is an improvement of val acc")
    parser.add_argument("--predict", action='store_true', help="This option needs to be called only if we want to predict and not train the model. Specify the model to load with --load_name")
    parser.add_argument("--load_name", type=str, default="simple_one_layer", help="Name of the model to load when using --predict")
    parser.add_argument("--overwrite_cached_data", action='store_true', help="Enable this option if we want to force again preprocessing and discard previous cached files")
    parser.add_argument("--force_shuffle", action='store_true', help="This options enables the shuffling of training and validation data if the user wants more control between runs.")
    parser.add_argument("--cluster_username", type=str, help="Use this option if you want to save models in the cluster by specifying your username which will be used to save in the scratch folder.")
    parser.add_argument("--fix_shuffle", action='store_true', help="This option allows to always shuffle the data in the same way by setting a random seed, ensuring reproducibility between runs")
    parser.add_argument("--step_size", type=int, default=1024, help="Steps size for training. Default is 1024. Compatible onyl with Bert models")
    args = parser.parse_args()


    # Check if running in the cluster
    if (args.cluster_username is not None):
        constants.SCRATCH_FOLDER = constants.SCRATCH_FOLDER / args.cluster_username
        constants.SAVED_MODELS_FOLDER = constants.SCRATCH_FOLDER / constants.SAVED_MODELS_FOLDER_CLUSTER
        constants.SAVED_MODELS_FOLDER_FROM_MODELS = constants.SCRATCH_FOLDER / constants.SAVED_MODELS_FOLDER_CLUSTER


    # PreProcess Data
    print("\nStarting Pre-Processing Data")
    input_data = run_preprocessing(args)
    if (args.force_shuffle):
        input_data = reshuffle(input_data, fixed = args.fix_shuffle)
    if ("elmo" in args.model_type):
        input_data = preprocess_data_elmo(args, input_data)

    # If using loaded model
    if (args.predict):
        load_path = Path(constants.SAVED_MODELS_FOLDER / str(args.model_type) / str(args.load_name))
        print("Loading Model " + str(load_path))

        if ("elmo" in args.model_type):
            input_data.batch_size = 16 #hacky way for ELMO to work (knows bug of ELMO). Doesn't change results
            args.batch_size = 16 #hacky way for ELMO to work (knows bug of ELMO). Doesn't change results
            reconstructed_model = match_model(args, input_data)
            reconstructed_model.load_weights(load_path)
            print("Model Loaded, starting predictions (ELMO Embedding)")
            predictions = reconstructed_model.predict(np.array(input_data.test_data), batch_size=16, verbose = 2)
        elif ("bert" in args.model_type):
            predictions = predict_bert(args, input_data, load_path)
        else:
            reconstructed_model = load_model(load_path)
            print("Model Loaded, starting predictions")
            predictions = reconstructed_model.predict(np.array(input_data.test_data), verbose = 2)
        print("Saving Preidctions")
        save_predictions(predictions, args)
        exit()

    print(f"\nRunning Analysis with model {args.model_type}, {args.dataset_size} dataset, {args.number_epochs} epochs and batch size of {args.batch_size}.\n")

    # Match argument model with known model and run it
    model, predictions = match_model(args, input_data)

    # Save Predictions done in a CSV with correct naming
    print("Saving Preidctions")
    save_predictions(predictions, args)


if __name__ == "__main__":
    main()
