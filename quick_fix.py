# QUICK FIX: Add these lines at the very beginning of your training script/notebook

import tensorflow as tf
import os

# Fix 1: Enable eager execution and configure TensorFlow
print("Configuring TensorFlow...")

# Enable eager execution if not already enabled
if not tf.executing_eagerly():
    tf.config.experimental_run_functions_eagerly(True)

# Set up GPU memory growth (prevents memory issues)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup warning: {e}")

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution enabled: {tf.executing_eagerly()}")

# Fix 2: Replace your PyDataset class with this fixed version
# If you're using a custom PyDataset, update it like this:

class FixedPyDataset(tf.keras.utils.PyDataset):
    def __init__(self, x_data, y_data, batch_size=32, **kwargs):
        # This fixes the warning about super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
    
    def __len__(self):
        return int(len(self.x_data) // self.batch_size)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return self.x_data[start:end], self.y_data[start:end]

# Fix 3: If the error persists, replace your model.fit() call with this:
"""
# Original problematic code:
# history1 = model.fit(train_gen, validation_data=val_gen, ...)

# Fixed version:
try:
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=create_callbacks(),
        verbose=1
    )
except Exception as e:
    print(f"Training error: {e}")
    print("Trying with eager execution forced...")
    
    # Force eager execution for this operation
    with tf.device('/CPU:0'):  # Use CPU to avoid GPU graph mode issues
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=create_callbacks(),
            verbose=1
        )
"""

print("Quick fix applied! Run your training code now.")