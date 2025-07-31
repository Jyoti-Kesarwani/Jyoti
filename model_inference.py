import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from enhanced_deepfake_detection import preprocess_face, focal_loss, create_vit_classifier

class DeepfakeDetector:
    """
    A unified deepfake detector that can load and use any of the trained models.
    """
    
    def __init__(self, model_path, model_type='auto'):
        """
        Initialize the detector with a trained model.
        
        Args:
            model_path: Path to the saved model file
            model_type: Type of model ('efficientnet', 'xception', 'vit', or 'auto')
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            if self.model_type == 'vit' or 'vit' in self.model_path.lower():
                # For ViT models, we need to recreate the architecture
                self.model = create_vit_classifier()
                self.model.load_weights(self.model_path)
            else:
                # For EfficientNet and Xception models
                custom_objects = {'focal_loss_fixed': focal_loss()}
                self.model = load_model(self.model_path, custom_objects=custom_objects)
            
            print(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Extract face from image
        face = preprocess_face(image_path, target_size=(224, 224))
        
        if face is None:
            raise ValueError(f"Could not process image: {image_path}")
        
        # Normalize pixel values
        face = face.astype(np.float32) / 255.0
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def predict_single(self, image_path, return_confidence=True):
        """
        Predict whether a single image is real or fake.
        
        Args:
            image_path: Path to the image file
            return_confidence: Whether to return confidence scores
            
        Returns:
            Prediction result (dict or string)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Extract probabilities
            fake_prob = float(prediction[0][0])
            real_prob = float(prediction[0][1])
            
            # Determine class
            predicted_class = "Real" if real_prob > fake_prob else "Fake"
            confidence = max(real_prob, fake_prob)
            
            if return_confidence:
                return {
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'real_probability': real_prob,
                    'fake_probability': fake_prob
                }
            else:
                return predicted_class
                
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths, return_confidence=True):
        """
        Predict multiple images at once.
        
        Args:
            image_paths: List of image file paths
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_single(image_path, return_confidence)
            results.append({
                'image_path': image_path,
                'result': result
            })
        
        return results
    
    def evaluate_directory(self, directory_path, true_label=None):
        """
        Evaluate all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            true_label: True label for the images ('Real' or 'Fake')
            
        Returns:
            Evaluation results
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        # Collect all image files
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(directory_path, file))
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        # Make predictions
        predictions = self.predict_batch(image_paths, return_confidence=True)
        
        # Calculate statistics
        real_count = sum(1 for p in predictions if p['result'] and p['result']['prediction'] == 'Real')
        fake_count = len(predictions) - real_count
        
        results = {
            'total_images': len(predictions),
            'predicted_real': real_count,
            'predicted_fake': fake_count,
            'predictions': predictions
        }
        
        # If true label is provided, calculate accuracy
        if true_label:
            correct_predictions = sum(1 for p in predictions 
                                    if p['result'] and p['result']['prediction'] == true_label)
            accuracy = correct_predictions / len(predictions) if predictions else 0
            results['accuracy'] = accuracy
            results['correct_predictions'] = correct_predictions
        
        return results

def compare_models(image_path, model_paths):
    """
    Compare predictions from multiple models on the same image.
    
    Args:
        image_path: Path to the image to analyze
        model_paths: Dictionary of model names and their file paths
        
    Returns:
        Comparison results
    """
    results = {}
    
    for model_name, model_path in model_paths.items():
        try:
            detector = DeepfakeDetector(model_path)
            result = detector.predict_single(image_path)
            results[model_name] = result
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = None
    
    return results

def main_demo():
    """
    Demonstration of the inference functionality.
    """
    print("=" * 60)
    print("DEEPFAKE DETECTION INFERENCE DEMO")
    print("=" * 60)
    
    # Example usage - update these paths as needed
    model_paths = {
        'EfficientNetB7': 'best_model_EfficientNetB7_auc.h5',
        'Xception': 'best_model_Xception_auc.h5',
        'ViT': 'best_model_ViT_auc.h5'
    }
    
    # Test single image prediction
    test_image = "path/to/test/image.jpg"  # Update this path
    
    if os.path.exists(test_image):
        print(f"\nTesting image: {test_image}")
        print("-" * 40)
        
        comparison_results = compare_models(test_image, model_paths)
        
        for model_name, result in comparison_results.items():
            if result:
                print(f"{model_name:15} | {result['prediction']:4} | "
                      f"Confidence: {result['confidence']:.3f}")
            else:
                print(f"{model_name:15} | ERROR")
    
    # Test directory evaluation
    test_directory = "path/to/test/directory"  # Update this path
    
    if os.path.exists(test_directory):
        print(f"\nEvaluating directory: {test_directory}")
        print("-" * 40)
        
        # Use the first available model
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                detector = DeepfakeDetector(model_path)
                results = detector.evaluate_directory(test_directory)
                
                print(f"Results using {model_name}:")
                print(f"Total images: {results['total_images']}")
                print(f"Predicted Real: {results['predicted_real']}")
                print(f"Predicted Fake: {results['predicted_fake']}")
                break

if __name__ == "__main__":
    main_demo()