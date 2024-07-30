# model

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

with open('../Stacked-LSTM/qa.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Encode the labels (answers) to numeric values
label_encoder = LabelEncoder()
df['encoded_answer'] = label_encoder.fit_transform(df['answer'])

# Prepare the dataset
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
input_ids = []
attention_masks = []

for question in df['question']:
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)
labels = tf.constant(df['encoded_answer'].values)

# Split the dataset into training and test sets
train_inputs_idx, test_inputs_idx, train_masks_idx, test_masks_idx, train_labels_idx, test_labels_idx = train_test_split(
    range(len(input_ids)), range(len(attention_masks)), range(len(labels)), test_size=0.2, random_state=42
)

# Convert to tensors
train_inputs = tf.gather(input_ids, train_inputs_idx)
test_inputs = tf.gather(input_ids, test_inputs_idx)
train_masks = tf.gather(attention_masks, train_masks_idx)
test_masks = tf.gather(attention_masks, test_masks_idx)
train_labels = tf.gather(labels, train_labels_idx)
test_labels = tf.gather(labels, test_labels_idx)