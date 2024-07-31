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