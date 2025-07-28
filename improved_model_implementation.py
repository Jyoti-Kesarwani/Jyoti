"""
IMMEDIATE IMPROVEMENTS FOR YOUR EXISTING MODEL
==============================================

This file shows exactly how to modify your current code for better accuracy.
Simply replace your existing code with these improved versions.
"""

import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
import numpy as np

# ============================================================================
# IMPROVED LOSS FUNCTIONS
# ============================================================================

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss - better for imbalanced datasets
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fixed

def label_smoothing_loss(smoothing=0.1):
    """
    Label smoothing to prevent overconfidence
    """
    def loss_fn(y_true, y_pred):
        y_true = y_true * (1 - smoothing) + smoothing / 2
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return loss_fn

# ============================================================================
# REPLACE YOUR EXISTING MODEL.COMPILE() WITH THIS:
# ============================================================================

# OPTION 1: Quick improvements to your existing code
model.compile(
    # AdamW is often better than Adam
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5  # L2 regularization
    ),
    
    # Use focal loss for imbalanced data, or keep binary_crossentropy
    loss=focal_loss(alpha=0.25, gamma=2.0),  # or 'binary_crossentropy'
    
    # Add F1Score for better evaluation
    metrics=[
        AUC(name='auc'), 
        Precision(name='precision'), 
        Recall(name='recall'), 
        'accuracy',
        tf.keras.metrics.F1Score(name='f1_score')
    ]
)

# ============================================================================
# REPLACE YOUR EXISTING CALLBACKS WITH THESE:
# ============================================================================

# Enhanced callbacks
callbacks = [
    # Monitor AUC instead of loss, increase patience
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',  # Better metric than val_loss
        patience=10,        # Increased from 5
        restore_best_weights=True,
        mode='max',         # Since we want to maximize AUC
        verbose=1
    ),
    
    # More aggressive learning rate reduction
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,         # More aggressive than 0.5
        patience=5,         # Increased from 3
        min_lr=1e-7,        # Lower minimum
        verbose=1
    ),
    
    # Save best model based on AUC
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_auc.h5', 
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Add cosine annealing learning rate
    tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + np.cos(np.pi * epoch / 50)),
        verbose=1
    )
]

# ============================================================================
# REPLACE YOUR EXISTING MODEL.FIT() WITH THIS:
# ============================================================================

# Enhanced training
history = model.fit(
    train_generator,
    epochs=50,          # Increased from 25 (early stopping will handle overfitting)
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks
)

# ============================================================================
# IMPROVED DATA AUGMENTATION (if using images)
# ============================================================================

def create_enhanced_data_generators():
    """
    More aggressive data augmentation
    """
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
    
    # Validation data should only be rescaled
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

# ============================================================================
# TRANSFER LEARNING MODEL (RECOMMENDED)
# ============================================================================

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Use pre-trained model for much better accuracy
    """
    # Use EfficientNetB0 (good balance of accuracy and speed)
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom head
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
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model, base_model

# ============================================================================
# TWO-PHASE TRAINING STRATEGY
# ============================================================================

def two_phase_training(model, base_model, train_gen, val_gen, class_weight):
    """
    First train with frozen backbone, then fine-tune
    """
    
    # Phase 1: Train with frozen backbone
    print("Phase 1: Training with frozen backbone...")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),  # Higher LR
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=[AUC(name='auc'), Precision(), Recall(), 'accuracy']
    )
    
    history1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    # Phase 2: Unfreeze and fine-tune
    print("Phase 2: Fine-tuning with unfrozen layers...")
    base_model.trainable = True
    
    # Use much lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),  # Much lower LR
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=[AUC(name='auc'), Precision(), Recall(), 'accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks  # Use the enhanced callbacks from above
    )
    
    return history1, history2

# ============================================================================
# COMPLETE EXAMPLE USAGE
# ============================================================================

def complete_improved_example():
    """
    Complete example showing all improvements together
    """
    
    # 1. Create improved model (if starting fresh)
    input_shape = (224, 224, 3)  # Adjust for your data
    model, base_model = create_transfer_learning_model(input_shape)
    
    # 2. Enhanced data generators
    train_datagen, val_datagen = create_enhanced_data_generators()
    
    # 3. Two-phase training
    history1, history2 = two_phase_training(
        model, base_model, train_generator, val_generator, class_weight
    )
    
    return model, history1, history2

# ============================================================================
# ENSEMBLE APPROACH (ADVANCED)
# ============================================================================

def create_ensemble_predictions(models_list, test_data):
    """
    Combine predictions from multiple models
    """
    predictions = []
    
    for model in models_list:
        pred = model.predict(test_data)
        predictions.append(pred)
    
    # Average predictions (or use weighted average)
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Or use majority voting for classification
    # ensemble_pred = (np.mean(predictions, axis=0) > 0.5).astype(int)
    
    return ensemble_pred

# ============================================================================
# QUICK IMPLEMENTATION CHECKLIST
# ============================================================================

"""
IMPLEMENTATION CHECKLIST - Apply these changes in order:

✅ IMMEDIATE (5 minutes):
1. Replace optimizer with AdamW
2. Change EarlyStopping monitor to 'val_auc' and increase patience to 10
3. Add F1Score to metrics
4. Increase epochs to 50 (early stopping will handle overfitting)

✅ QUICK (15 minutes):
5. Implement focal loss
6. Add cosine annealing learning rate scheduler
7. Enhance data augmentation parameters
8. Add ModelCheckpoint monitoring 'val_auc'

✅ MEDIUM (30 minutes):
9. Implement transfer learning with EfficientNetB0
10. Add two-phase training strategy
11. Add BatchNormalization layers

✅ ADVANCED (1+ hours):
12. Implement ensemble methods
13. Add cross-validation
14. Hyperparameter tuning with Keras Tuner

EXPECTED ACCURACY IMPROVEMENTS:
- Immediate changes: +2-5% accuracy
- Transfer learning: +5-15% accuracy  
- Full implementation: +10-25% accuracy
"""

# ============================================================================
# YOUR EXACT CODE - IMPROVED VERSION
# ============================================================================

"""
Replace your existing code with this improved version:
"""

# Improved compilation
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss=focal_loss(alpha=0.25, gamma=2.0),  # or keep 'binary_crossentropy'
    metrics=[AUC(name='auc'), Precision(name='precision'), Recall(name='recall'), 
             'accuracy', tf.keras.metrics.F1Score(name='f1_score')]
)

# Improved callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=10, restore_best_weights=True, mode='max', verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_auc.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1
    ),
    tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + np.cos(np.pi * epoch / 50))
    )
]

# Improved training
history = model.fit(
    train_generator,
    epochs=50,  # Increased epochs with better early stopping
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks
)