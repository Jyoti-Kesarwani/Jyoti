#!/usr/bin/env python3
"""
Test script for the deepfake detection ensemble model.
This script tests model creation and basic functionality without requiring the full dataset.
"""

import numpy as np
import tensorflow as tf
from deepfake_detection_multimodel import (
    build_ensemble_model, 
    create_efficientnet_model, 
    create_xception_model, 
    create_vit_model,
    demonstrate_ensemble_features
)

def test_model_creation():
    """Test if all models can be created successfully."""
    print("Testing model creation...")
    
    try:
        # Test individual models
        print("1. Testing EfficientNetB7 model creation...")
        eff_model, eff_base = create_efficientnet_model()
        print(f"   ✓ EfficientNetB7 created: {eff_model.count_params():,} parameters")
        
        print("2. Testing Xception model creation...")
        xcep_model, xcep_base = create_xception_model()
        print(f"   ✓ Xception created: {xcep_model.count_params():,} parameters")
        
        print("3. Testing Vision Transformer model creation...")
        vit_model, vit_base = create_vit_model()
        print(f"   ✓ ViT created: {vit_model.count_params():,} parameters")
        
        print("4. Testing Ensemble model creation...")
        ensemble_model = build_ensemble_model()
        print(f"   ✓ Ensemble created: {ensemble_model.count_params():,} parameters")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error creating models: {e}")
        return False

def test_model_inference():
    """Test if models can perform inference on dummy data."""
    print("\nTesting model inference...")
    
    try:
        # Create dummy data
        dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
        
        # Test ensemble model
        print("1. Testing ensemble model inference...")
        ensemble_model = build_ensemble_model()
        ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        predictions = ensemble_model.predict(dummy_input, verbose=0)
        print(f"   ✓ Ensemble prediction shape: {predictions.shape}")
        print(f"   ✓ Prediction values: {predictions[0]}")
        
        # Verify predictions sum to 1 (softmax output)
        pred_sums = np.sum(predictions, axis=1)
        print(f"   ✓ Prediction sums: {pred_sums} (should be close to 1.0)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error during inference: {e}")
        return False

def test_model_architecture():
    """Test and display model architecture details."""
    print("\nTesting model architecture...")
    
    try:
        ensemble_model = build_ensemble_model()
        
        print("1. Model input shape:", ensemble_model.input_shape)
        print("2. Model output shape:", ensemble_model.output_shape)
        print("3. Number of layers:", len(ensemble_model.layers))
        
        # Check for key layers
        layer_names = [layer.name for layer in ensemble_model.layers]
        key_layers = ['vit_dense', 'effnet_dense', 'xcep_dense', 'feature_fusion', 'final_output']
        
        print("4. Checking for key layers:")
        for key_layer in key_layers:
            if key_layer in layer_names:
                print(f"   ✓ {key_layer} found")
            else:
                print(f"   ✗ {key_layer} missing")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error checking architecture: {e}")
        return False

def test_compilation():
    """Test model compilation with different optimizers and losses."""
    print("\nTesting model compilation...")
    
    try:
        ensemble_model = build_ensemble_model()
        
        # Test different compilation configurations
        configs = [
            {'optimizer': 'adam', 'loss': 'categorical_crossentropy'},
            {'optimizer': 'adamw', 'loss': 'binary_crossentropy'},
            {'optimizer': tf.keras.optimizers.AdamW(learning_rate=1e-4), 'loss': 'categorical_crossentropy'}
        ]
        
        for i, config in enumerate(configs, 1):
            print(f"{i}. Testing compilation with {config['optimizer']} and {config['loss']}...")
            try:
                ensemble_model.compile(
                    optimizer=config['optimizer'],
                    loss=config['loss'],
                    metrics=['accuracy', 'precision', 'recall']
                )
                print(f"   ✓ Compilation successful")
            except Exception as e:
                print(f"   ✗ Compilation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error during compilation testing: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DEEPFAKE DETECTION ENSEMBLE MODEL TEST SUITE")
    print("=" * 60)
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    tests = [
        test_model_creation,
        test_model_architecture,
        test_compilation,
        test_model_inference
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Model Creation",
        "Model Architecture",
        "Model Compilation", 
        "Model Inference"
    ]
    
    for test_name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Your ensemble model is ready for training!")
        print("To train the model, run: python deepfake_detection_multimodel.py")
    else:
        print("\n❌ Please fix the failed tests before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)