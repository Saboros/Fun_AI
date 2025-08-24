import torch
import numpy as np
from fusion_preproces import FusionDataProcessor
from motion_classifier import MotionClassifier
from heartmodel import HeartRateAnomalyDetector

def test_data_shapes():
    """Test that data shapes are correct for the models"""
    print("=== Testing Data Shapes ===")
    
    # Initialize preprocessor
    preprocessor = FusionDataProcessor()
    
    # Generate small dataset for testing
    print("Generating test data...")
    motion_data, hr_data, labels = preprocessor.create_fusion_dataset(num_samples=100)
    
    # Preprocess data
    print("Preprocessing data...")
    motion_processed = preprocessor.preprocess_motion_data(motion_data)
    hr_processed = preprocessor.preprocess_hr_data(hr_data)
    
    print(f"Original motion data shape: {motion_data.shape}")
    print(f"Processed motion data shape: {motion_processed.shape}")
    print(f"Original HR data shape: {hr_data.shape}")
    print(f"Processed HR data shape: {hr_processed.shape}")
    
    # Create sequences
    print("Creating sequences...")
    X_motion, X_hr, y = preprocessor.create_fusion_sequences(
        motion_processed, hr_processed, labels
    )
    
    print(f"Final motion sequences shape: {X_motion.shape}")
    print(f"Final HR sequences shape: {X_hr.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Test motion classifier
    print("\n=== Testing Motion Classifier ===")
    motion_model = MotionClassifier()
    
    # Test with single sample
    test_motion = X_motion[:1]  # Take first sample
    print(f"Test motion input shape: {test_motion.shape}")
    
    try:
        with torch.no_grad():
            output = motion_model(test_motion)
            print(f"Motion model output shape: {output.shape}")
            print("✅ Motion classifier works!")
    except Exception as e:
        print(f"❌ Motion classifier error: {e}")
    
    # Test HR model
    print("\n=== Testing HR Model ===")
    hr_model = HeartRateAnomalyDetector()
    
    # Test with single sample
    test_hr = X_hr[:1]  # Take first sample
    print(f"Test HR input shape: {test_hr.shape}")
    
    try:
        with torch.no_grad():
            output = hr_model(test_hr)
            print(f"HR model output shape: {output.shape}")
            print("✅ HR model works!")
    except Exception as e:
        print(f"❌ HR model error: {e}")
    
    # Test fusion features
    print("\n=== Testing Fusion Features ===")
    motion_model.eval()
    hr_model.eval()
    
    try:
        with torch.no_grad():
            # Get motion predictions
            motion_outputs = motion_model(X_motion[:5])  # Test with 5 samples
            motion_probs = torch.softmax(motion_outputs, dim=1)
            print(f"Motion probabilities shape: {motion_probs.shape}")
            
            # Get HR predictions
            hr_outputs = hr_model(X_hr[:5])
            hr_probs = torch.sigmoid(hr_outputs)
            print(f"HR probabilities shape: {hr_probs.shape}")
            
            # Combine features
            fusion_features = torch.cat([motion_probs, hr_probs], dim=1)
            print(f"Fusion features shape: {fusion_features.shape}")
            print("✅ Fusion features creation works!")
            
    except Exception as e:
        print(f"❌ Fusion features error: {e}")

if __name__ == "__main__":
    test_data_shapes()
