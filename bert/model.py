

# Convert to tensors

# Convert to tf.data.Dataset


# Load model


# Custom train step

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
