import json
import os
import pickle

import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

path = "output_dir/"
try:
    os.makedirs(path)
except:
    pass

dataset = pd.read_csv('./dataset/clean_qa.txt', delimiter="|", header=None,lineterminator='\n')

dataset_val = dataset.iloc[1794:].to_csv('output_dir/val.csv')

dataset_train = dataset.iloc[:1794]

questions_train = dataset_train.iloc[:, 0].values.tolist()
answers_train = dataset_train.iloc[:, 1].values.tolist()

questions_test = dataset_train.iloc[:, 0].values.tolist()
answers_test = dataset_train.iloc[:, 1].values.tolist()

def save_tokenizer(tokenizer):
    with open('output_dir/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_config(key, value):
    data = {}
    if os.path.exists(path + 'config.json'):
        with open(path + 'config.json') as json_file:
            data = json.load(json_file)

    data[key] = value
    with open(path + 'config.json', 'w') as outfile:
        json.dump(data, outfile)

target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex, lower=True)
tokenizer.fit_on_texts(questions_train + answers_train + questions_test + answers_test)
save_tokenizer(tokenizer)
