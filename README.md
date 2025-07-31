# Enhanced Deepfake Detection System

A comprehensive deepfake detection system that implements and compares three state-of-the-art deep learning models:
- **EfficientNetB7**: Advanced convolutional neural network
- **Xception**: Depthwise separable convolutions architecture  
- **Vision Transformer (ViT)**: Transformer-based image classification

## Features

### 🎯 Multi-Model Architecture
- **EfficientNetB7**: Efficient scaling of CNN depth, width, and resolution
- **Xception**: Extreme version of Inception with depthwise separable convolutions
- **Vision Transformer**: Self-attention mechanism for image patches

### 🔧 Advanced Preprocessing
- Face detection using OpenCV Haar Cascades
- Automatic face extraction and cropping
- Fallback to center crop when face detection fails
- Enhanced data augmentation strategies

### 📊 Comprehensive Training Strategy
- Progressive training with transfer learning
- Two-stage training for CNN models (head training + fine-tuning)
- End-to-end training for Vision Transformer
- Class imbalance handling with focal loss
- Advanced callbacks and learning rate scheduling

### 📈 Evaluation & Visualization
- Detailed performance metrics (AUC, Precision, Recall, F1-Score)
- Confusion matrices and ROC curves for each model
- Training history visualization
- Model comparison dashboard
- Grad-CAM visualization for interpretability

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt

# Verify TensorFlow GPU installation (optional)
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## Dataset Structure

Organize your dataset in the following structure:
```
Celeb-DF Preprocessed/
├── Train/
│   ├── Real/
│   │   ├── image1.jpg
│   │   └── ...
│   └── Fake/
│       ├── image1.jpg
│       └── ...
├── Val/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

## Configuration

Update the `Config` class in `enhanced_deepfake_detection.py`:

```python
class Config:
    MAIN_PATH = r"path/to/your/dataset"  # Update this path
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 15
    FINE_TUNE_EPOCHS = 20
    INITIAL_LR = 1e-4
    FINE_TUNE_LR = 1e-5
```

## Usage

### Basic Training
```python
# Run the complete pipeline
from enhanced_deepfake_detection import main_pipeline

trained_models, models_results, comparison_results = main_pipeline()
```

### Individual Model Training
```python
# Train only specific models
from enhanced_deepfake_detection import *

# Prepare data
train_df, valid_df, test_df = prepare_datasets()
train_gen, val_gen, test_gen = create_data_generators(train_df, valid_df, test_df)

# Train EfficientNetB7
model, base_model = create_efficientnet_model()
trained_model, histories = progressive_training(model, base_model, train_gen, val_gen, class_weights, 'EfficientNetB7')

# Evaluate
y_pred, y_pred_proba, auc_score = evaluate_model(trained_model, test_gen, test_df, 'EfficientNetB7')
```

### Command Line Execution
```bash
python enhanced_deepfake_detection.py
```

## Model Architectures

### EfficientNetB7
- **Base**: EfficientNetB7 pretrained on ImageNet
- **Head**: Global Average Pooling → Dropout → Dense(512) → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Dense(2)
- **Training**: Two-stage (frozen backbone + fine-tuning)
- **Loss**: Focal Loss (stage 1) + Categorical Crossentropy (stage 2)

### Xception
- **Base**: Xception pretrained on ImageNet
- **Head**: Global Average Pooling → Dropout → Dense(512) → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Dense(2)
- **Training**: Two-stage (frozen backbone + fine-tuning)
- **Loss**: Focal Loss (stage 1) + Categorical Crossentropy (stage 2)

### Vision Transformer (ViT)
- **Architecture**: Custom ViT implementation
- **Patches**: 16x16 patches from 224x224 images
- **Layers**: 8 transformer layers with 4 attention heads
- **Head**: Global Average Pooling → Dense(2048) → Dense(1024) → Dense(2)
- **Training**: End-to-end training
- **Loss**: Categorical Crossentropy

## Performance Metrics

The system evaluates models using:
- **AUC Score**: Area Under the ROC Curve
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Accuracy**: Overall correctness

## Advanced Features

### Focal Loss
Handles class imbalance by focusing on hard-to-classify examples:
```python
focal_loss(gamma=2.0, alpha=0.25)
```

### Data Augmentation
Comprehensive augmentation strategy:
- Rotation (±30°)
- Width/Height shifts (±30%)
- Shear and zoom (±30%)
- Horizontal and vertical flips
- Brightness variation (0.7-1.3x)

### Learning Rate Scheduling
- Cosine annealing with warm restarts
- ReduceLROnPlateau for adaptive learning rate
- Early stopping based on validation AUC

### Model Interpretation
- Grad-CAM visualization for CNN models
- Attention map visualization for ViT
- Feature importance analysis

## Hyperparameter Tuning

Optional hyperparameter optimization using Keras Tuner:
```python
from enhanced_deepfake_detection import hyperparameter_tuning

best_model, tuner = hyperparameter_tuning(train_gen, val_gen)
```

## Model Comparison

The system automatically compares all models and provides:
- Performance ranking by AUC score
- Detailed metrics comparison
- Visual performance comparison charts

## Output Files

The training process generates:
- `Train.csv`, `Val.csv`, `Test.csv`: Dataset information
- `best_model_EfficientNetB7_auc.h5`: Best EfficientNetB7 model
- `best_model_Xception_auc.h5`: Best Xception model
- `best_model_ViT_auc.h5`: Best ViT model
- Training plots and evaluation charts

## System Requirements

### Minimum Requirements
- CPU: 4+ cores
- RAM: 16GB
- Storage: 10GB free space
- GPU: 4GB VRAM (optional but recommended)

### Recommended Requirements
- CPU: 8+ cores
- RAM: 32GB
- Storage: 20GB free space
- GPU: 8GB+ VRAM (RTX 3070 or better)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in Config
   - Use mixed precision training
   - Enable gradient checkpointing

2. **Face Detection Fails**
   - Check image quality and format
   - Ensure faces are clearly visible
   - Consider alternative face detection methods

3. **Training Slow**
   - Verify GPU usage
   - Reduce image size or model complexity
   - Use data parallel training

### Performance Optimization

1. **Mixed Precision Training**
```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

2. **Multi-GPU Training**
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{enhanced_deepfake_detection,
  title={Enhanced Deepfake Detection with Multi-Model Comparison},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/enhanced-deepfake-detection}
}
```

## Acknowledgments

- EfficientNet paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- Xception paper: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- Vision Transformer paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- Celeb-DF dataset: [The DeepFake Detection Challenge (DFDC) Dataset](https://arxiv.org/abs/2006.07397)