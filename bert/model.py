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

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs, train_masks), train_labels)).shuffle(len(train_labels)).batch(150)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_inputs, test_masks), test_labels)).batch(150)

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(label_encoder.classes_))

# Custom train step
@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
    logits = model(x, training=True).logits
    loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training loop
epochs = 400
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train)
        if step % 50 == 0:
            print(f"Training loss (for one batch) at step {step}: {loss:.4f}")

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')
