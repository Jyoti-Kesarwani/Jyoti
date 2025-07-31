import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import keras_tuner as kt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enhanced configuration
class Config:
    MAIN_PATH = r"C:\Users\Jyoti\Downloads\archive\Celeb-DF Preprocessed"
    IMAGE_SIZE = (224, 224)  # Increased from 128x128
    BATCH_SIZE = 32  # Reduced for better gradient updates
    INITIAL_EPOCHS = 15
    FINE_TUNE_EPOCHS = 20
    INITIAL_LR = 1e-4
    FINE_TUNE_LR = 1e-5
    VIT_IMAGE_SIZE = (224, 224)  # ViT specific image size

# Enhanced data preprocessing with face detection
def preprocess_face(image_path, target_size=(224, 224)):
    """Extract and preprocess face from image."""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first detected face
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)
            return face
        else:
            # Fallback to center crop if no face detected
            return cv2.resize(img, target_size)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Data generation and analysis
def create_dataframe(main_path, sub_dir):
    """Create DataFrame with file paths and labels."""
    data = {"file_path": [], "label": []}
    for label_dir, label in zip(['Real', 'Fake'], [1, 0]):
        folder_path = os.path.join(main_path, sub_dir, label_dir)
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, img_file)
                # Verify image can be loaded
                if preprocess_face(file_path) is not None:
                    data["file_path"].append(file_path)
                    data["label"].append(label)
    return pd.DataFrame(data)

def analyze_dataset(df):
    """Analyze dataset distribution."""
    print("Dataset Analysis:")
    print("=" * 50)
    print("Class distribution:")
    print(df['label'].value_counts())
    print(f"Total samples: {len(df)}")

# Generate datasets
def prepare_datasets():
    """Prepare train, validation, and test datasets."""
    sub_dirs = ['Train', 'Test', 'Val']
    
    for sub_dir in sub_dirs:
        df = create_dataframe(Config.MAIN_PATH, sub_dir)
        csv_path = f"{sub_dir}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} with {len(df)} entries.")
    
    # Load and shuffle datasets
    train_df = pd.read_csv("Train.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = pd.read_csv("Val.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv("Test.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convert labels to strings for categorical mode
    for df in [train_df, valid_df, test_df]:
        df['label'] = df['label'].astype(str)
    
    # Analyze datasets
    print("\nTraining Set:")
    analyze_dataset(train_df)
    print("\nValidation Set:")
    analyze_dataset(valid_df)
    print("\nTest Set:")
    analyze_dataset(test_df)
    
    return train_df, valid_df, test_df

# Enhanced data augmentation
def create_data_generators(train_df, valid_df, test_df):
    """Create enhanced data generators with better augmentation."""
    
    # Enhanced training augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,      # Increased from typical 20
        width_shift_range=0.3,  # Increased
        height_shift_range=0.3, # Increased
        shear_range=0.3,        # Increased
        zoom_range=0.3,         # Increased
        horizontal_flip=True,
        vertical_flip=True,     # Add if appropriate for your data
        brightness_range=[0.7, 1.3],  # Add brightness variation
        fill_mode='nearest'
    )
    
    # Validation and test generators (no augmentation)
    datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file_path',
        y_col='label',
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical'
    )
    
    valid_generator = datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='file_path',
        y_col='label',
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='file_path',
        y_col='label',
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator

# Focal loss for handling class imbalance
def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function to handle class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Vision Transformer Implementation
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(input_shape=(224, 224, 3), patch_size=16, num_patches=196, 
                         projection_dim=64, num_heads=4, transformer_units=[128, 64],
                         transformer_layers=8, mlp_head_units=[2048, 1024]):
    """Create Vision Transformer model."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Create patches
    patches = PatchExtractor(patch_size)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = tf.keras.layers.Dense(transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.1)(x3)
        x3 = tf.keras.layers.Dense(transformer_units[1], activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        encoded_patches = tf.keras.layers.Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.GlobalAveragePooling1D()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = representation
    for units in mlp_head_units:
        features = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(features)
        features = tf.keras.layers.Dropout(0.5)(features)
    
    # Classify outputs
    logits = tf.keras.layers.Dense(2, activation='softmax')(features)
    
    # Create the Keras model
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

# Model creation functions
def create_efficientnet_model(input_shape=(224, 224, 3)):
    """Create improved EfficientNetB7 model."""
    base_model = EfficientNetB7(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Initially freeze base model
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    return model, base_model

def create_xception_model(input_shape=(224, 224, 3)):
    """Create Xception model."""
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Initially freeze base model
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    return model, base_model

def create_vit_model(input_shape=(224, 224, 3)):
    """Create Vision Transformer model."""
    model = create_vit_classifier(input_shape=input_shape)
    return model, None  # ViT doesn't have a separate base model

# Learning rate scheduling
def lr_schedule(epoch):
    initial_lr = 1e-4
    decay = 0.95
    new_lr = initial_lr * (decay ** epoch)
    return float(new_lr)

# Enhanced callbacks
def create_callbacks(model_name):
    """Create comprehensive callbacks."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=10, restore_best_weights=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_model_{model_name}_auc.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + np.cos(np.pi * epoch / 50))
        )
    ]
    return callbacks

# Progressive training strategy
def progressive_training(model, base_model, train_gen, val_gen, class_weights, model_name):
    """Implement progressive training strategy."""
    
    print("=" * 60)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("=" * 60)
    
    if base_model is not None:  # For EfficientNet and Xception
        print("STAGE 1: Training classifier head only")
        print("=" * 60)
        
        # Stage 1: Train only classifier head
        base_model.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
            loss=focal_loss(alpha=0.25, gamma=2.0),
            metrics=[AUC(name='auc'), Precision(name='precision'), Recall(name='recall'), 
                     'accuracy', tf.keras.metrics.F1Score(name='f1_score')]
        )
        
        model.summary()
        
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=create_callbacks(model_name),
            verbose=1
        )
        
        print("=" * 60)
        print("STAGE 2: Fine-tuning last layers")
        print("=" * 60)
        
        # Stage 2: Fine-tune last layers
        base_model.trainable = True
        
        # Freeze early layers, unfreeze last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model.compile(
            optimizer=Adam(learning_rate=Config.FINE_TUNE_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
        )
        
        print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
        
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=create_callbacks(model_name),
            verbose=1
        )
        
        return model, [history1, history2]
    
    else:  # For ViT
        print("Training Vision Transformer end-to-end")
        print("=" * 60)
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
            loss='categorical_crossentropy',
            metrics=[AUC(name='auc'), Precision(name='precision'), Recall(name='recall'), 
                     'accuracy', tf.keras.metrics.F1Score(name='f1_score')]
        )
        
        model.summary()
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.INITIAL_EPOCHS + Config.FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=create_callbacks(model_name),
            verbose=1
        )
        
        return model, [history]

# Model evaluation and visualization
def evaluate_model(model, test_gen, test_df, model_name):
    """Comprehensive model evaluation."""
    print("=" * 60)
    print(f"{model_name.upper()} MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    y_pred_proba = predictions[:, 1]  # Probability of being real
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return y_pred, y_pred_proba, roc_auc

def plot_training_history(histories, model_name):
    """Plot training history."""
    if len(histories) == 2:  # Two-stage training
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['accuracy', 'loss']
        
        for i, metric in enumerate(metrics):
            for j, (history, stage) in enumerate(zip(histories, ['Stage 1', 'Stage 2'])):
                if metric in history.history:
                    axes[i, j].plot(history.history[metric], label=f'Training {metric}')
                    axes[i, j].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                    axes[i, j].set_title(f'{model_name} {stage} - {metric.capitalize()}')
                    axes[i, j].set_xlabel('Epoch')
                    axes[i, j].set_ylabel(metric.capitalize())
                    axes[i, j].legend()
    else:  # Single-stage training
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        metrics = ['accuracy', 'loss']
        history = histories[0]
        
        for i, metric in enumerate(metrics):
            if metric in history.history:
                axes[i].plot(history.history[metric], label=f'Training {metric}')
                axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                axes[i].set_title(f'{model_name} - {metric.capitalize()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# Model comparison function
def compare_models(models_results):
    """Compare performance of all models."""
    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    comparison_data = []
    for model_name, (y_pred, y_pred_proba, auc_score) in models_results.items():
        comparison_data.append({
            'Model': model_name,
            'AUC Score': auc_score
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC Score', ascending=False)
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_df, x='Model', y='AUC Score')
    plt.title('Model Performance Comparison (AUC Score)')
    plt.ylabel('AUC Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# Main training and evaluation pipeline
def main_pipeline():
    """Main pipeline for training and evaluating all models."""
    print("=" * 80)
    print("DEEPFAKE DETECTION - MULTI-MODEL COMPARISON")
    print("=" * 80)
    
    # Prepare datasets
    train_df, valid_df, test_df = prepare_datasets()
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(train_df, valid_df, test_df)
    
    # Calculate class weights
    class_counts = train_df['label'].value_counts()
    total_samples = len(train_df)
    class_weights = {
        0: total_samples / (2 * class_counts['0']),  # Fake
        1: total_samples / (2 * class_counts['1'])   # Real
    }
    
    print(f"Class weights: {class_weights}")
    
    # Define models to train
    models_to_train = {
        'EfficientNetB7': create_efficientnet_model,
        'Xception': create_xception_model,
        'ViT': create_vit_model
    }
    
    trained_models = {}
    models_results = {}
    
    # Train and evaluate each model
    for model_name, model_func in models_to_train.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING {model_name.upper()}")
        print(f"{'='*80}")
        
        # Create model
        model, base_model = model_func()
        
        # Train model
        trained_model, histories = progressive_training(
            model, base_model, train_gen, val_gen, class_weights, model_name
        )
        
        # Plot training history
        plot_training_history(histories, model_name)
        
        # Evaluate model
        y_pred, y_pred_proba, auc_score = evaluate_model(
            trained_model, test_gen, test_df, model_name
        )
        
        # Store results
        trained_models[model_name] = trained_model
        models_results[model_name] = (y_pred, y_pred_proba, auc_score)
    
    # Compare all models
    comparison_results = compare_models(models_results)
    
    return trained_models, models_results, comparison_results

# Grad-CAM for model interpretation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

if __name__ == "__main__":
    # Run the main pipeline
    trained_models, models_results, comparison_results = main_pipeline()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print("Best performing model:", comparison_results.iloc[0]['Model'])
    print("Best AUC score:", comparison_results.iloc[0]['AUC Score'])