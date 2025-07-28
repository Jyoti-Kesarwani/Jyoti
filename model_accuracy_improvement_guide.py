"""
Deep Learning Model Accuracy Improvement Techniques
==================================================

This guide provides comprehensive strategies to improve your model's accuracy
based on your current binary classification setup.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.metrics import AUC, Precision, Recall
import numpy as np

# ============================================================================
# 1. DATA-RELATED IMPROVEMENTS
# ============================================================================

def improve_data_quality():
    """
    Data quality is often the most impactful factor for model accuracy
    """
    
    # Data Augmentation for images
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1)
    ])
    
    # For text data - consider techniques like:
    # - Synonym replacement
    # - Back translation
    # - Paraphrasing
    
    return data_augmentation

def create_improved_data_generators():
    """
    Enhanced data generators with better preprocessing
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        # Add validation split if not done separately
        validation_split=0.2
    )
    
    return train_datagen

# ============================================================================
# 2. MODEL ARCHITECTURE IMPROVEMENTS
# ============================================================================

def create_improved_model(input_shape, num_classes=1):
    """
    Enhanced model architecture with proven techniques
    """
    model = models.Sequential()
    
    # Use pre-trained backbone (Transfer Learning)
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Fine-tune only top layers initially
    base_model.trainable = False
    
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    
    # Add regularization layers
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    # Output layer
    if num_classes == 1:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model, base_model

def create_residual_block(x, filters, kernel_size=3, stride=1):
    """
    Add residual connections to prevent vanishing gradients
    """
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

# ============================================================================
# 3. ADVANCED OPTIMIZATION TECHNIQUES
# ============================================================================

def get_improved_optimizer_and_scheduler():
    """
    Advanced optimization strategies
    """
    
    # 1. AdamW optimizer (often better than Adam)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    # 2. Cosine Annealing Learning Rate Schedule
    def cosine_annealing_schedule(epoch, lr):
        max_lr = 1e-3
        min_lr = 1e-6
        cycle_length = 20
        
        cycle_progress = epoch % cycle_length
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * cycle_progress / cycle_length))
    
    lr_scheduler = callbacks.LearningRateScheduler(cosine_annealing_schedule)
    
    # 3. Warm restart
    warm_restart = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    return optimizer, [lr_scheduler, warm_restart]

# ============================================================================
# 4. ADVANCED CALLBACKS AND TRAINING STRATEGIES
# ============================================================================

def get_advanced_callbacks():
    """
    Comprehensive callback setup for better training
    """
    callbacks_list = [
        # Early stopping with more patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Model checkpoint with multiple metrics
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_auc.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Custom callback for gradual unfreezing
        GradualUnfreezing(start_epoch=10, layers_per_epoch=5)
    ]
    
    return callbacks_list

class GradualUnfreezing(tf.keras.callbacks.Callback):
    """
    Gradually unfreeze layers during training
    """
    def __init__(self, start_epoch=10, layers_per_epoch=5):
        super().__init__()
        self.start_epoch = start_epoch
        self.layers_per_epoch = layers_per_epoch
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            # Unfreeze layers gradually
            layers_to_unfreeze = min(
                (epoch - self.start_epoch) * self.layers_per_epoch,
                len(self.model.layers)
            )
            
            for i, layer in enumerate(self.model.layers):
                if i >= len(self.model.layers) - layers_to_unfreeze:
                    layer.trainable = True

# ============================================================================
# 5. LOSS FUNCTION IMPROVEMENTS
# ============================================================================

def get_improved_loss_functions():
    """
    Advanced loss functions for better training
    """
    
    # 1. Focal Loss for imbalanced datasets
    def focal_loss(alpha=0.25, gamma=2.0):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fixed
    
    # 2. Label Smoothing
    def label_smoothing_loss(smoothing=0.1):
        def loss_fn(y_true, y_pred):
            y_true = y_true * (1 - smoothing) + smoothing / 2
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return loss_fn
    
    return focal_loss(), label_smoothing_loss()

# ============================================================================
# 6. ENSEMBLE METHODS
# ============================================================================

def create_ensemble_model(input_shape, num_models=5):
    """
    Create an ensemble of models for better accuracy
    """
    models_list = []
    
    for i in range(num_models):
        model = models.Sequential([
            # Vary architecture slightly for each model
            layers.Dense(512 + i*64, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3 + i*0.1),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        models_list.append(model)
    
    return models_list

def ensemble_predict(models_list, X):
    """
    Make predictions using ensemble
    """
    predictions = []
    for model in models_list:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# ============================================================================
# 7. IMPROVED TRAINING PROCEDURE
# ============================================================================

def improved_training_procedure(model, train_gen, val_gen, class_weight):
    """
    Complete improved training procedure
    """
    
    # Phase 1: Train with frozen base model
    print("Phase 1: Training with frozen base model...")
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
        loss=get_improved_loss_functions()[0],  # Focal loss
        metrics=[AUC(name='auc'), Precision(name='precision'), 
                Recall(name='recall'), 'accuracy']
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
    
    # Phase 2: Fine-tune with unfrozen layers
    print("Phase 2: Fine-tuning with unfrozen layers...")
    
    # Unfreeze base model
    if hasattr(model, 'layers') and len(model.layers) > 0:
        if hasattr(model.layers[0], 'trainable'):
            model.layers[0].trainable = True
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
        loss=get_improved_loss_functions()[1],  # Label smoothing
        metrics=[AUC(name='auc'), Precision(name='precision'), 
                Recall(name='recall'), 'accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=get_advanced_callbacks()
    )
    
    return history1, history2

# ============================================================================
# 8. CROSS-VALIDATION FOR ROBUST EVALUATION
# ============================================================================

def k_fold_cross_validation(X, y, k=5):
    """
    Implement k-fold cross-validation
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{k}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train model
        model, _ = create_improved_model(X.shape[1:])
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC()]
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # Evaluate
        val_score = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(val_score[1])  # accuracy
    
    return cv_scores

# ============================================================================
# 9. HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning_example():
    """
    Example of hyperparameter tuning using Keras Tuner
    """
    
    try:
        import keras_tuner as kt
        
        def build_model(hp):
            model = models.Sequential()
            
            # Tune the number of layers and units
            for i in range(hp.Int('num_layers', 2, 5)):
                model.add(layers.Dense(
                    units=hp.Int(f'units_{i}', 32, 512, step=32),
                    activation='relu'
                ))
                model.add(layers.Dropout(
                    hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)
                ))
            
            model.add(layers.Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        tuner = kt.RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=50
        )
        
        return tuner
    
    except ImportError:
        print("Install keras-tuner for hyperparameter tuning: pip install keras-tuner")
        return None

# ============================================================================
# 10. PRACTICAL APPLICATION TO YOUR MODEL
# ============================================================================

def apply_improvements_to_your_model():
    """
    Direct improvements you can apply to your existing code
    """
    
    # Your improved model compilation
    model.compile(
        # Try AdamW instead of Adam
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5
        ),
        
        # Try focal loss for imbalanced data
        loss=focal_loss(alpha=0.25, gamma=2.0),
        
        # Add more comprehensive metrics
        metrics=[
            AUC(name='auc'), 
            Precision(name='precision'), 
            Recall(name='recall'), 
            'accuracy',
            tf.keras.metrics.F1Score(name='f1_score')
        ]
    )
    
    # Enhanced callbacks
    callbacks = [
        # Longer patience for early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',  # Monitor AUC instead of loss
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        
        # More aggressive learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7
        ),
        
        # Save best model based on AUC
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_auc.h5', 
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        
        # Cosine annealing
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + np.cos(np.pi * epoch / 50))
        )
    ]
    
    # Enhanced training
    history = model.fit(
        train_generator,
        epochs=50,  # More epochs with better callbacks
        validation_data=val_generator,
        class_weight=class_weight,
        callbacks=callbacks
    )
    
    return history

# ============================================================================
# SUMMARY OF KEY RECOMMENDATIONS
# ============================================================================

"""
TOP 10 IMMEDIATE IMPROVEMENTS FOR YOUR MODEL:

1. **Use Transfer Learning**: Replace your model with a pre-trained backbone
2. **Advanced Optimizer**: Switch from Adam to AdamW
3. **Better Loss Function**: Use Focal Loss for imbalanced data
4. **Enhanced Callbacks**: Monitor AUC instead of loss, increase patience
5. **Data Augmentation**: Add more diverse augmentation techniques
6. **Regularization**: Add BatchNormalization and adjust dropout rates
7. **Learning Rate Scheduling**: Implement cosine annealing or cyclical LR
8. **Gradual Unfreezing**: Start with frozen backbone, then gradually unfreeze
9. **Cross-Validation**: Use k-fold CV for robust model evaluation
10. **Ensemble Methods**: Train multiple models and average predictions

QUICK WINS (implement first):
- Change optimizer to AdamW
- Monitor 'val_auc' instead of 'val_loss' in callbacks
- Increase EarlyStopping patience to 10
- Add F1Score to metrics
- Use more aggressive data augmentation

MEDIUM-TERM IMPROVEMENTS:
- Implement transfer learning with EfficientNet or ResNet
- Try focal loss or label smoothing
- Add learning rate scheduling
- Implement gradual unfreezing strategy

ADVANCED TECHNIQUES:
- Build ensemble models
- Implement custom training loops
- Use advanced architectures (attention mechanisms, etc.)
- Hyperparameter tuning with Keras Tuner
"""