


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

path = "output_dir_14k/"
os.makedirs(path, exist_ok=True)

dataset = pd.read_csv('data.csv', delimiter="|", header=None, lineterminator='\n')

















tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)
maxlen_answers_train = max([len(x) for x in tokenized_answers_train]) if tokenized_answers_train else 0
save_config('maxlen_answers', maxlen_answers_train)
decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')

tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)
maxlen_answers_test = max([len(x) for x in tokenized_answers_test]) if tokenized_answers_test else 0
save_config('maxlen_answers', maxlen_answers_test)
decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')

for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:]
padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')
decoder_output_data_train = to_categorical(padded_answers_train, num_classes=VOCAB_SIZE)

for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]
padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')
decoder_output_data_test = to_categorical(padded_answers_test, num_classes=VOCAB_SIZE)

enc_inp = Input(shape=(None,))
enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inp)

# Stacked LSTM for Encoder
enc_lstm1 = LSTM(256, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
enc_lstm2 = LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5)

enc_output1, enc_state_h1, enc_state_c1 = enc_lstm1(enc_embedding)
enc_output2, enc_state_h2, enc_state_c2 = enc_lstm2(enc_output1)
enc_states = [enc_state_h2, enc_state_c2]

dec_inp = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inp)

# Stacked LSTM for Decoder
dec_lstm1 = LSTM(256, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
dec_lstm2 = LSTM(256, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)

dec_output1, _, _ = dec_lstm1(dec_embedding, initial_state=enc_states)
dec_output2, _, _ = dec_lstm2(dec_output1)

dec_dense = Dense(VOCAB_SIZE, activation=softmax)
output = dec_dense(dec_output2)

logdir = os.path.join(path, "logs")
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

checkpoint = ModelCheckpoint(
    os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    save_freq=150  # Assuming 'train_data' is your training dataset
)

model = Model([enc_inp, dec_inp], output)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 100
epochs = 1000
model.fit([encoder_input_data_train, decoder_input_data_train],
          decoder_output_data_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
          callbacks=[tensorboard_callback, checkpoint])
model.save(os.path.join(path, 'model-' + path.replace("/", "") + '.h5'))
