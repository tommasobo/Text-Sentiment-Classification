## Saved Models

Here we provide the download link to some pre-loaded models which can be quickly use to make predictions. Note that some models are quite big files (>1GB) and can take some time to make predictions if not using a GPU.

- [Basic NN](https://polybox.ethz.ch/index.php/s/WYWqaU6WqsTKLtS) is the model we used for our Basic Neural Network baseline. Please add the downloaded model inside this folder (create it if not existing) `data/saved_models/basic_nn/`. Then make predictions by using `python3 main.py basic_nn --fix_shuffle --predict --load_name="full_epoch01_valacc:0.82.hdf5"`

- [Bi-LSTM](https://polybox.ethz.ch/index.php/s/2wOTI3imbQn7BWh) is our model using Keras Embedding and a bidirectional LSTM architecture. Please add the downloaded model inside this folder (create it if not existing) `data/saved_models/bi_lstm/`. Then make predictions by using `python3 main.py bi_lstm --fix_shuffle --predict --load_name="full_epoch03_valacc:0.85.hdf5"`

- [ELMo Bi-GRU](https://polybox.ethz.ch/index.php/s/TfYsW1l3IRHh4jc) is our model using ELMo Embedding and a bidirectional GRU architecture. Please add the downloaded model inside this folder (create it if not existing) `data/saved_models/elmo_bigru/`. Then make predictions by using `python3 main.py elmo_bigru --fix_shuffle --predict --load_name="batch_size256_datafull_epoch05_.hdf5"`

- [ELMo Bi-LSTM](https://polybox.ethz.ch/index.php/s/NzgHjJjSGZeAap0) is our model using ELMo Embedding and a bidirectional LSTM architecture. Please add the downloaded model inside this folder (create it if not existing) `data/saved_models/elmo_bilstm_3/`. Then make predictions by using `python3 main.py elmo_bilstm_3 --fix_shuffle --predict --load_name="batch_size128_datafull_epoch07_.hdf5"`

- [ELMo Bi-LSTM with MHAT](https://polybox.ethz.ch/index.php/s/8RZh4fOkd9mmI8C) is our model using ELMo Embedding, a bidirectional LSTM architecture and a Multi Head attention layer. Please add the downloaded model inside this folder (create it if not existing) `data/saved_models/elmo_bilstm_multi_attention/`. Then make predictions by using `python3 main.py elmo_bilstm_multi_attention --fix_shuffle --predict --load_name="batch_size256_datafull_epoch08_.hdf5"`

- [BERT](https://polybox.ethz.ch/index.php/s/fgZHXbk4RfqkCsF) is our model using BERT Embedding and a simple fully connected netowrk. Please add the downloaded model inside this folder (create it if not existing) `data/saved_models/elmo_bilstm_multi_attention/`. Then make predictions by using `python3 main.py bert_6 --fix_shuffle --predict --load_name="batch_size16_datafull_steps2000_epoch64.hdf5"`