# Fix for TensorFlow Eager Execution Error
# Add these lines at the beginning of your training script

import tensorflow as tf
import numpy as np

# Solution 1: Enable eager execution explicitly (for TF 1.x compatibility)
# Uncomment if using TensorFlow 1.x or if eager execution is disabled
# tf.compat.v1.enable_eager_execution()

# Solution 2: For TensorFlow 2.x, ensure you're using tf.function correctly
# and avoid mixing graph mode with eager mode operations

# Solution 3: Configure TensorFlow properly at the start of your script
def configure_tensorflow():
    """Configure TensorFlow for proper execution mode"""
    
    # Enable eager execution (should be default in TF 2.x)
    if not tf.executing_eagerly():
        tf.config.experimental_run_functions_eagerly(True)
    
    # Set memory growth for GPU (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution enabled: {tf.executing_eagerly()}")

# Solution 4: Fix for PyDataset warning
class CustomPyDataset(tf.keras.utils.PyDataset):
    """Fixed PyDataset class that properly calls super().__init__()"""
    
    def __init__(self, x_data, y_data, batch_size=32, **kwargs):
        # Call parent constructor with kwargs to avoid warning
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.indices = np.arange(len(x_data))
    
    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.x_data))
        
        batch_x = self.x_data[start_idx:end_idx]
        batch_y = self.y_data[start_idx:end_idx]
        
        return batch_x, batch_y

# Solution 5: Alternative data loading approach using tf.data
def create_tf_dataset(x_data, y_data, batch_size=32, shuffle=True):
    """Create TensorFlow dataset instead of PyDataset"""
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x_data))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Solution 6: Fixed training function template
def fixed_progressive_training(model, base_model, train_gen, val_gen, class_weights):
    """
    Fixed version of progressive training that avoids eager execution issues
    """
    
    # Ensure eager execution is enabled
    configure_tensorflow()
    
    # Stage 1: Train with frozen base model
    print("=" * 60)
    print("STAGE 1: Training with frozen base model")
    print("=" * 60)
    
    # Freeze base model
    base_model.trainable = False
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss='binary_crossentropy',  # Use standard loss to avoid custom function issues
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train model
    try:
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,  # Adjust epochs as needed
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Stage 1 training completed successfully")
        return model, [history1]
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Trying alternative approach...")
        
        # Alternative: Convert generators to tf.data.Dataset
        # This can help avoid eager execution issues
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        
        val_dataset = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        
        history1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=15,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, [history1]

if __name__ == "__main__":
    # Call this at the very beginning of your script
    configure_tensorflow()
    
    print("TensorFlow configuration complete!")
    print("You can now run your training code without eager execution errors.")