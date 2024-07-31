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

VOCAB_SIZE = len(tokenizer.word_index) + 1
save_config('VOCAB_SIZE', VOCAB_SIZE)
print('Vocabulary size : {}'.format(VOCAB_SIZE))

tokenized_questions_train = tokenizer.texts_to_sequences(questions_train)
maxlen_questions_train = max([len(x) for x in tokenized_questions_train])
save_config('maxlen_questions', maxlen_questions_train)
encoder_input_data_train = pad_sequences(tokenized_questions_train, maxlen=maxlen_questions_train, padding='post')

tokenized_questions_test = tokenizer.texts_to_sequences(questions_test)
maxlen_questions_test = max([len(x) for x in tokenized_questions_test])
save_config('maxlen_questions', maxlen_questions_test)
encoder_input_data_test = pad_sequences(tokenized_questions_test, maxlen=maxlen_questions_test, padding='post')

tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)
maxlen_answers_train = max([len(x) for x in tokenized_answers_train])
save_config('maxlen_answers', maxlen_answers_train)
decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')

tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)
maxlen_answers_test = max([len(x) for x in tokenized_answers_test])
save_config('maxlen_answers', maxlen_answers_test)
decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')