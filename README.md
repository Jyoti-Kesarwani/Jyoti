# Deepfake Detection Multi-Model System

A comprehensive deepfake detection system that implements and compares multiple state-of-the-art architectures including EfficientNetB7, Xception, Vision Transformer (ViT), and an innovative ensemble model that combines all three approaches.

## 🚀 Features

### Model Architectures
- **EfficientNetB7**: State-of-the-art CNN with compound scaling for optimal accuracy-efficiency trade-off
- **Xception**: Depthwise separable convolutions for computational efficiency
- **Vision Transformer (ViT)**: Self-attention mechanism for capturing global image features
- **Ensemble Model**: Novel combination of all three architectures for maximum performance

### Advanced Training Features
- **Progressive Training Strategy**: Two-stage training for CNN models (feature extraction → fine-tuning)
- **Focal Loss**: Handles class imbalance effectively
- **Advanced Data Augmentation**: Comprehensive image transformations for robustness
- **Face Detection**: Automatic face extraction using OpenCV Haar cascades
- **Class Weight Balancing**: Automatic calculation for imbalanced datasets

### Evaluation & Analysis
- **Comprehensive Metrics**: AUC, Precision, Recall, F1-Score, Accuracy
- **Visualization**: ROC curves, confusion matrices, training history plots
- **Model Comparison**: Side-by-side performance analysis
- **Grad-CAM**: Model interpretability for CNN architectures

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM recommended
- 50GB+ free disk space for datasets

### Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Dataset Structure

The system expects the Celeb-DF dataset with the following structure:
```
Celeb-DF Preprocessed/
├── Train/
│   ├── Real/
│   └── Fake/
├── Val/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

## 🔧 Configuration

Update the `Config` class in the main script to match your setup:

```python
class Config:
    MAIN_PATH = "path/to/your/Celeb-DF Preprocessed"  # Update this path
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 15
    FINE_TUNE_EPOCHS = 20
    ENSEMBLE_EPOCHS = 25
```

## 🚀 Usage

### Basic Usage
```python
from deepfake_detection_multimodel import main_pipeline

# Run the complete pipeline
trained_models, results, comparison = main_pipeline()
```

### Individual Model Training
```python
from deepfake_detection_multimodel import *

# Prepare data
train_df, valid_df, test_df = prepare_datasets()
train_gen, val_gen, test_gen = create_data_generators(train_df, valid_df, test_df)

# Train specific model
model, base_model = create_efficientnet_model()
trained_model, histories = progressive_training(model, base_model, train_gen, val_gen, class_weights, "EfficientNetB7")
```

### Ensemble Model Only
```python
# Create and train ensemble model
ensemble_model = build_ensemble_model(input_shape=(224, 224, 3), num_classes=2)
trained_ensemble, histories = progressive_training(ensemble_model, None, train_gen, val_gen, class_weights, "Ensemble")
```

## 🏗️ Model Architecture Details

### EfficientNetB7
- **Base**: Pre-trained on ImageNet
- **Input**: 224×224×3 images
- **Features**: Compound scaling, inverted residual blocks
- **Training**: Progressive (frozen → fine-tuned)

### Xception
- **Base**: Pre-trained on ImageNet
- **Input**: 224×224×3 images
- **Features**: Depthwise separable convolutions
- **Training**: Progressive (frozen → fine-tuned)

### Vision Transformer (ViT)
- **Patch Size**: 16×16 pixels
- **Projection Dim**: 64
- **Transformer Layers**: 8
- **Multi-head Attention**: 4 heads
- **Training**: End-to-end

### Ensemble Model
- **Architecture**: Three-branch parallel processing
- **Feature Fusion**: Concatenated features from all models
- **Processing**: Individual dense layers for each branch
- **Classification**: Shared dense layers with dropout and batch normalization

## 📊 Performance Metrics

The system evaluates models using:
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Correct predictions / Total predictions

## 🔍 Model Interpretability

### Grad-CAM Support
For CNN models (EfficientNetB7, Xception), the system includes Grad-CAM visualization:

```python
# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_array, model, 'last_conv_layer_name')
```

## 📈 Training Strategy

### Progressive Training (CNN Models)
1. **Stage 1**: Train only classifier head with frozen base model
   - Learning Rate: 1e-4
   - Loss: Focal Loss
   - Epochs: 15

2. **Stage 2**: Fine-tune last 20 layers
   - Learning Rate: 1e-5
   - Loss: Categorical Crossentropy
   - Epochs: 20

### End-to-End Training (ViT & Ensemble)
- Learning Rate: 1e-4 with weight decay
- Loss: Categorical Crossentropy
- Optimizer: AdamW

## 🎯 Key Innovations

1. **Multi-Architecture Ensemble**: First implementation combining EfficientNet, Xception, and ViT
2. **Progressive Feature Fusion**: Individual processing before concatenation
3. **Focal Loss Integration**: Better handling of class imbalance
4. **Comprehensive Evaluation**: Multiple metrics and visualization tools
5. **Face-Centric Preprocessing**: Automatic face detection and cropping

## 📝 Results Format

The system outputs:
- Trained models dictionary
- Performance results (predictions, probabilities, AUC scores)
- Comparison DataFrame with rankings
- Visualization plots (confusion matrices, ROC curves, training history)

## 🛠️ Customization

### Adding New Models
```python
def create_new_model(input_shape=(224, 224, 3)):
    # Your model implementation
    return model, base_model

# Add to models_to_train dictionary
models_to_train['NewModel'] = create_new_model
```

### Custom Loss Functions
```python
def custom_loss(y_true, y_pred):
    # Your loss implementation
    return loss_value

# Use in model compilation
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
```

## 📞 Support

For issues or questions:
1. Check the error logs for specific issues
2. Ensure dataset structure matches expected format
3. Verify GPU memory availability for large models
4. Adjust batch size if encountering memory issues

## 🔮 Future Enhancements

- Support for additional datasets (FaceForensics++, DFDC)
- Real-time video processing capabilities
- Advanced ensemble techniques (stacking, voting)
- Mobile-optimized model variants
- API endpoint for deployment

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with dataset licenses and ethical AI guidelines when using for commercial applications.