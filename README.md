# Text Sentiment Classification

The use of microblogging and text messaging as a media of communication has greatly increased over the past 10 years. Such large volumes of data amplifies the need for automatic methods to understand the opinion conveyed in a text.
In this project we have to classify whether tweets had originally a positive smiley ":)" or a negative one ":(".

## Summary of Results
In order to do so we tried several different approaches including differen type of embeddings (Keras, word2Vec, GloVe, ELMo and BERT) and different type of neural networks (Bi-LSTM with attention, CNN, Fully connected...). Overall with the final use of ensemble models, we were able to achieve an accuracy of above 90%.

## Installation Instructions

The following steps will describe what is necessary to run the software both on a local machine and on the ETH Leonhard cluster after downloading the repository.

### Local Installation

Please make sure that you are running Python >= 3.6 (note we didn't test compatibility with Python >= 3.8).Also if installing the GPU version make sure to have the right version of cuDNN and CUDA. For more info [click here](https://www.tensorflow.org/install/source#linux).

To run the program we strongly recommend using a virtual environment in order to have an easier time with package dependencies. In order to do so run this commands to create a virtual environment.
```bash
pip3 install virtualenv
virtualenv normal_env
source normal_env/bin/activate
```

Once inside the virtual environment please install all the dependencies using the installation file. If you get an error during installation please ignore it, it is not relevant for the project.

```bash
pip3 install -r requirements_local.txt
```

Note that in order to run models with ELMO it is necessary to install an older version of tensorflow since TF 2.0 and newer aren't compatible with ELMO (https://github.com/tensorflow/hub/issues/412). In order to do so we recommend using another virtual environment for this specific purpose in order to being able to quickly switch between the twos.

```bash
virtualenv elmoenv
source elmoenv/bin/activate
pip3 install -r requirements_elmo_local.txt
```


### Cluster Installation

To run the code on the cluster we first need to load the correct module. When executing the program with the basic models excluding ElMo we need to load the following modules.
```bash
module load gcc/6.3.0 python_gpu/3.7.4
module load eth_proxy
```

If we want to use ElMo models we need an older verison of tensorflow, so use the following modules.
```bash
module load gcc/4.8.5 python_gpu/3.7.1
module load eth_proxy
```

After that we need to install (for the first run only) requirements on the cluster with the installation file. Note that this is enough for both ElMo and non-ElMo versions.
```bash
pip install --user -r requirements_cluster.txt
```
Please note that using virtual environment on the cluster resulted in compatibility problems that aren't easy to fix without having privileges on the cluster so we recommend not using them.

If you are still getting problems on the cluster please delete all the previous python packages installed by running the following command and then redo the process described above.
```bash
rm -rf /cluster/home/ethzusername/.local/lib/python3.7/site-packages/*
```
### Donwload Dataset

In order to use the program both locally or on the cluster we first need to download the dataset (if not provided with the git repository). Note: you must be connected to the ETH VPN to download the dataset and give permissions to the script. Also to run the self-trained GloVe baseline on a non-unix system, you must re-run data_download.sh after manually extracting the twitter dataset.

Overall to download the whole dataset necessary for the project run
```bash
chmod u+x data_download.sh
./data_download.sh
```

**Using flag parameters**: Without arguments it just downloads the necessary input data. With arguments, in addition to downloading the data it will:

if -c local:

create and activate a virtual environment using virtualenv, then install requirements_local.txt or requirements_elmo_local.txt depending on whether the elmo argument (-e) is present.

if -c cluster:

install requirements_cluster.txt (note that you must load the appropriate modules manually as outlined in the Cluster installation section)

**Usage**: data_download.sh -c [local/cluster] -e

If you are still having issues please manually download all the input files inside the data/input folder.

## How to Use

In order to train one of the models use the `main.py` script:
```bash
usage: main.py [-h] [--dataset_size DATASET_SIZE]
               [--number_epochs NUMBER_EPOCHS] [--batch_size BATCH_SIZE]
               [--save_model] [--glove_model GLOVE_MODEL]
               [--no_save_best_epochs] [--predict] [--load_name LOAD_NAME]
               [--overwrite_cached_data] [--disable_train_val_shuffle]
               [--cluster_username CLUSTER_USERNAME] [--fix_shuffle]
               [--step_size STEP_SIZE]
               model_type

Run Sentimental Analysis

positional arguments:
  model_type            Name of the model to run

optional arguments:
  -h, --help            show this help message and exit
  --dataset_size DATASET_SIZE
                        Which dataset to use (options are small or full). Default is full
  --number_epochs NUMBER_EPOCHS
                        Number of epochs. Default is 3
  --batch_size BATCH_SIZE
                        Batch size for training. Default is 128
  --save_model          Use this option to save the NN model
  --glove_model GLOVE_MODEL
                        Only relevant for glove model. Whether we want to used
                        pretrained glove or train it ourselves (options are
                        learned or pretrained). Default is learned
  --no_save_best_epochs
                        Use this option to disable saving the full model after
                        each epoch where there is an improvement of val acc
  --predict             This option needs to be called only if we want to
                        predict and not train the model. Specify the model to
                        load with --load_name
  --load_name LOAD_NAME
                        Name of the model to load when using --predict
  --overwrite_cached_data
                        Enable this option if we want to force again
                        preprocessing and discard previous cached files
  --force_shuffle
                        This options enables the shuffling of training and
                        validation data if the user wants more control between
                        runs.
  --cluster_username CLUSTER_USERNAME
                        Use this option if you want to save models in the
                        cluster by specifying your username which will be used
                        to save in the scratch folder.
  --fix_shuffle         This option allows to always shuffle the data in the
                        same way by setting a random seed, ensuring
                        reproducibility between runs
  --step_size STEP_SIZE
                        Steps size for training. Default is 1024. Compatible
                        onyl with Bert models
```

### Models Available
This is the list of models we discuss in the paper. Please use one of these names when training or loading a model. Note that a more extensive list of models (including some not cited in the paper) is available inside `constants.py`.

Baselines:
```
[baseline_glove, baseline_glove_50, baseline_word2vec, basic_nn]
```

Keras:
```
[bi_lstm, simple_cnn, cnn_rnn, simple_rnn]
```

ELMo:
```
[elmo_lstm, elmo_bigru, elmo_bilstm_3, elmo_bilstm_multi_attention]
```

Bert:
```
[bert_4, bert_5, bert_6]
```

### Ensemble
To run the ensemble model please run `ensemble.py` and follow the instructions in the terminal. 

### Examples

To train a model ("basic_nn" for example) launch:
```bash
python3 main.py basic_nn --number_epochs=3 --batch_size=256 --dataset_size=full
```

To load a previously saved model launch:
```bash
python3 main.py basic_nn --predict --load_name="example.hdf5"
```

In both cases predictions will be saved inside the `data/output` folder

### Baselines
To run the glove baseline, use model baseline_glove or baseline_glove_50 (for 200 or 50 dimensional embeddings) and "--glove_model pretrained" or "--glove_model learned", to use either a pretrained GloVe model, or to train one on the training dataset.
Ensure you have run data_download.sh and have glove.twitter.27B.200d.txt and glove.twitter.27B.50d.txt in the data/input/ directory to use the pretrained model.
If not on a unix OS, you must re-run data_download.sh after extracting the twitter dataset to use the self-trained models.

The word2vec embedding can be run in the same way as other models, set baseline_word2vec as model_type and set your desired dataset size (other arguments are irrelevant). The required stopwords data is downloaded when running the code by default. Please Download the stopwords data manually by command "python -m nltk.downloader stopwords" if it is not found. 

## Reproducibility

In order to reproduce results that we obtained we recommend running the program with the `--fix_shuffle` option in order to set all the possible random seeds. If using a GPU (like we did) it is possible that some non-deterministic functions are still present as stated in this [Github issue](https://github.com/tensorflow/tensorflow/issues/2732). Regardless, based on our experience, the difference is somewhat minimal (<0.1%).
It is also recommended to use the `--overwrite_cached_data` option at least one time in order to delete previously cached input files (that were generated without the random seeds set). 

We also provide some of the best models that we used in the paper as models to donwload and load. Please [click here](data/saved_models/README.md) to get to the downlaod list.

Finally we provide a [link](https://polybox.ethz.ch/index.php/s/xhg5n9ad0zLhtDK) to the submission files that were used for the final ensemble run.

## Troubleshooting

**Q: I am getting an error while running the scripts locally. What could it be?**

A: The most likely answer is that you are not using the correct version of tensorflow. Please follow the installation instructions above and make sure you change environment when running elmo models.


**Q: I am getting an error while running the sripts on Leonhard. What could it be?**

A: The most likely answer is that you are not using the correct version of tensorflow. Please follow the installation instructions above and make sure you load the correct modules when using ElMo or non-ElMo models. If you still get errors we recommend removing everything inside your personal python folder and reinstalling everything.


**Q: I am getting an error similiar to "AttributeError: 'Bidirectional' object has no attribute 'outbound_nodes'" when running a specific model, what could it be?**

A: This is usually due to some weird interaction between Keras and TF Keras versions. Please make sure the correct versions are installed. If the issue is persistent try to temporarily comment line 3 of `main.py (os.environ['TF_KERAS'] = '1')`. 


**Q: I don't have a GPU, can I still run your models?**

A: Yes but you need to change the version of tensorflow from tensorflow-gpu to tensorflow in the requirement files. Also performance will be much slower with a CPU.


**Q: I am getting some error about libcud not loading or missing cuda modules.**

A: Please make sure to have donwloaded the right version of cuDNN and CUDA if using tensorflow-gpu. For more info [click here](https://www.tensorflow.org/install/source#linux).


**Q: How can I reproduce your exacts results?**

A: We tried setting all random seeds when possible in our code so just re-running everything from scratch should result in very similiar results (if not identical, but some randomness is still expected due to using the GPU version of TensorFlow). Otherwise we provide a download link to our best models so the user can simply load them into the program and make predictions. We also provide a link to all the submissions we used for the final ensemble model.


**Q: How can I disable saving the model after epoch run to save space?**

A: Models are saved after each run if the validation accuracy improves. It is possible to disable this behaviour by running the `no_save_best_epochs` flag when starting the program.


## Folder Structure

- `/data`: Contains all data files
	
	- `/output`: contains data generated from our model to be submitted. Note that each model has its own subdirectory with all submissions

  - `/input`: contains input data to be used for training and testing

  - `/saved_models`: NN models saved so they can be reused at a later stage. Subfolders organized by model name.

  - `/pre_processing_*`: contains preprocessed cached data.

  - `/ensemble`: contains submissions to be used for the ensemble model and its output.

- `/media`: for plots and report

- `/src`: code folder

  - `/models`: code for each specific preprocessing and neural network should go here