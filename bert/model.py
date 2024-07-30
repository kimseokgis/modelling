

# Convert to tensors

# Convert to tf.data.Dataset


# Load model


# Custom train step

    

# Compile the model


# Training loop

        loss = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train)
        if step % 50 == 0:
            print(f"Training loss (for one batch) at step {step}: {loss:.4f}")

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')
