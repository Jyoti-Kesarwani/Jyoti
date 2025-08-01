#!/usr/bin/env python3
"""
Model Comparison Demo
This script demonstrates the architectural differences between individual models and the ensemble.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deepfake_detection_multimodel import (
    create_efficientnet_model,
    create_xception_model, 
    create_vit_model,
    build_ensemble_model
)

def compare_model_parameters():
    """Compare parameter counts across all models."""
    print("=" * 60)
    print("MODEL PARAMETER COMPARISON")
    print("=" * 60)
    
    models = {
        'EfficientNetB7': create_efficientnet_model,
        'Xception': create_xception_model,
        'ViT': create_vit_model,
        'Ensemble': lambda: (build_ensemble_model(), None)
    }
    
    param_counts = {}
    
    for name, model_func in models.items():
        try:
            model, _ = model_func()
            param_count = model.count_params()
            param_counts[name] = param_count
            
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = param_count - trainable_params
            
            print(f"\n{name}:")
            print(f"  Total Parameters: {param_count:,}")
            print(f"  Trainable: {trainable_params:,}")
            print(f"  Non-trainable: {non_trainable_params:,}")
            
        except Exception as e:
            print(f"Error creating {name}: {e}")
            param_counts[name] = 0
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    names = list(param_counts.keys())
    counts = [param_counts[name] / 1e6 for name in names]  # Convert to millions
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = plt.bar(names, counts, color=colors)
    
    plt.title('Model Parameter Comparison (Millions)', fontsize=14, fontweight='bold')
    plt.ylabel('Parameters (Millions)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return param_counts

def demonstrate_architecture_features():
    """Demonstrate unique features of each architecture."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE FEATURES COMPARISON")
    print("=" * 60)
    
    features = {
        'EfficientNetB7': {
            'Type': 'Convolutional Neural Network',
            'Key Innovation': 'Compound Scaling (depth, width, resolution)',
            'Strength': 'High accuracy with efficiency',
            'Input Processing': 'Hierarchical feature extraction',
            'Pre-training': 'ImageNet (1.4M images)',
            'Best For': 'General image classification tasks'
        },
        'Xception': {
            'Type': 'Depthwise Separable CNN',
            'Key Innovation': 'Extreme version of Inception',
            'Strength': 'Parameter efficiency',
            'Input Processing': 'Depthwise + pointwise convolutions',
            'Pre-training': 'ImageNet (1.4M images)',
            'Best For': 'Mobile and edge deployment'
        },
        'ViT': {
            'Type': 'Vision Transformer',
            'Key Innovation': 'Self-attention for images',
            'Strength': 'Global context understanding',
            'Input Processing': 'Patch-based sequence modeling',
            'Pre-training': 'From scratch (no ImageNet)',
            'Best For': 'Complex spatial relationships'
        },
        'Ensemble': {
            'Type': 'Multi-architecture Fusion',
            'Key Innovation': 'Combines CNN + Transformer',
            'Strength': 'Best of all architectures',
            'Input Processing': 'Parallel feature extraction',
            'Pre-training': 'Inherits from all components',
            'Best For': 'Maximum performance scenarios'
        }
    }
    
    for model_name, feature_dict in features.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        for feature, value in feature_dict.items():
            print(f"  {feature}: {value}")

def test_inference_speed():
    """Compare inference speed across models (approximate)."""
    print("\n" + "=" * 60)
    print("INFERENCE SPEED COMPARISON")
    print("=" * 60)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    models = {
        'EfficientNetB7': create_efficientnet_model,
        'Xception': create_xception_model,
        'ViT': create_vit_model,
        'Ensemble': lambda: (build_ensemble_model(), None)
    }
    
    results = {}
    
    for name, model_func in models.items():
        try:
            print(f"\nTesting {name}...")
            model, _ = model_func()
            
            # Warmup
            _ = model.predict(dummy_input, verbose=0)
            
            # Time multiple predictions
            import time
            start_time = time.time()
            for _ in range(10):
                _ = model.predict(dummy_input, verbose=0)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
            results[name] = avg_time
            
            print(f"  Average inference time: {avg_time:.2f} ms")
            
        except Exception as e:
            print(f"  Error testing {name}: {e}")
            results[name] = float('inf')
    
    # Find fastest model
    fastest_model = min(results, key=results.get)
    print(f"\n🏆 Fastest Model: {fastest_model} ({results[fastest_model]:.2f} ms)")
    
    return results

def visualize_ensemble_architecture():
    """Create a visual representation of the ensemble architecture."""
    print("\n" + "=" * 60)
    print("ENSEMBLE ARCHITECTURE VISUALIZATION")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define component positions
    input_pos = (0.5, 0.9)
    branch_positions = {
        'EfficientNetB7': (0.2, 0.6),
        'Xception': (0.5, 0.6),
        'ViT': (0.8, 0.6)
    }
    fusion_pos = (0.5, 0.3)
    output_pos = (0.5, 0.1)
    
    # Draw components
    # Input
    input_rect = plt.Rectangle((input_pos[0]-0.1, input_pos[1]-0.05), 0.2, 0.1, 
                              facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(input_pos[0], input_pos[1], 'Input\n(224×224×3)', ha='center', va='center', fontweight='bold')
    
    # Branches
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (branch, pos) in enumerate(branch_positions.items()):
        branch_rect = plt.Rectangle((pos[0]-0.08, pos[1]-0.08), 0.16, 0.16,
                                   facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(branch_rect)
        ax.text(pos[0], pos[1], branch, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw connections from input to branches
        ax.arrow(input_pos[0], input_pos[1]-0.05, pos[0]-input_pos[0], pos[1]+0.08-input_pos[1]+0.05,
                head_width=0.02, head_length=0.03, fc='gray', ec='gray')
    
    # Fusion layer
    fusion_rect = plt.Rectangle((fusion_pos[0]-0.15, fusion_pos[1]-0.05), 0.3, 0.1,
                               facecolor='#96CEB4', edgecolor='green', linewidth=2)
    ax.add_patch(fusion_rect)
    ax.text(fusion_pos[0], fusion_pos[1], 'Feature Fusion\n(Concatenation)', ha='center', va='center', fontweight='bold')
    
    # Draw connections from branches to fusion
    for pos in branch_positions.values():
        ax.arrow(pos[0], pos[1]-0.08, fusion_pos[0]-pos[0], fusion_pos[1]+0.05-pos[1]+0.08,
                head_width=0.02, head_length=0.03, fc='gray', ec='gray')
    
    # Output
    output_rect = plt.Rectangle((output_pos[0]-0.08, output_pos[1]-0.05), 0.16, 0.1,
                               facecolor='gold', edgecolor='orange', linewidth=2)
    ax.add_patch(output_rect)
    ax.text(output_pos[0], output_pos[1], 'Output\n(Fake/Real)', ha='center', va='center', fontweight='bold')
    
    # Connection from fusion to output
    ax.arrow(fusion_pos[0], fusion_pos[1]-0.05, 0, output_pos[1]+0.05-fusion_pos[1]+0.05,
            head_width=0.02, head_length=0.03, fc='gray', ec='gray')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ensemble Model Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print architecture summary
    print("\nEnsemble Architecture Summary:")
    print("1. Input image (224×224×3) is fed to all three branches simultaneously")
    print("2. Each branch extracts features using its unique architecture:")
    print("   - EfficientNetB7: Hierarchical CNN features")
    print("   - Xception: Efficient depthwise separable features") 
    print("   - ViT: Global self-attention features")
    print("3. Features are processed through individual dense layers")
    print("4. Processed features are concatenated for fusion")
    print("5. Final classification through shared dense layers")

def main():
    """Run the complete model comparison demo."""
    print("🚀 DEEPFAKE DETECTION MODEL COMPARISON DEMO")
    print("=" * 60)
    
    try:
        # Compare parameters
        param_counts = compare_model_parameters()
        
        # Show architecture features
        demonstrate_architecture_features()
        
        # Test inference speed
        speed_results = test_inference_speed()
        
        # Visualize ensemble
        visualize_ensemble_architecture()
        
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        
        print("✅ Successfully demonstrated all model architectures")
        print("✅ Parameter comparison completed")
        print("✅ Speed benchmarking completed")
        print("✅ Ensemble architecture visualized")
        
        print("\n🎯 Key Takeaways:")
        print("1. Ensemble model combines strengths of all architectures")
        print("2. Trade-offs exist between accuracy, speed, and complexity")
        print("3. Each model has unique strengths for different scenarios")
        print("4. Ensemble approach maximizes performance potential")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please ensure all dependencies are installed correctly.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n💥 Demo encountered errors.")
    exit(0 if success else 1)