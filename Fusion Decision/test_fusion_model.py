import torch
import numpy as np
from model import FusionClassifier
from fusion_preproces import FusionDataProcessor
import warnings
warnings.filterwarnings("ignore")


def test_fusion_model():
    """Test the refactored fusion model"""
    print("=== Testing Refactored Fusion Model ===")
    
    # Initialize components
    fusion_model = FusionClassifier()
    preprocessor = FusionDataProcessor()
    
    # Generate sample data
    print("\nGenerating sample data...")
    motion_data, hr_data, labels = preprocessor.create_fusion_dataset(num_samples=100)
    
    # Preprocess data
    motion_processed = preprocessor.preprocess_motion_data(motion_data[:10])  # Use first 10 samples
    hr_processed = preprocessor.preprocess_hr_data(hr_data[:10])
    
    print(f"Motion data shape: {motion_processed.shape}")
    print(f"HR data shape: {hr_processed.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = fusion_model(motion_processed, hr_processed)
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")
    
    # Test prediction with probabilities
    print("\nTesting prediction with probabilities...")
    with torch.no_grad():
        logits = fusion_model(motion_processed, hr_processed)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Test environmental risk calculation
    print("\nTesting environmental risk calculation...")
    # Sample environmental data: [wave_height, current_strength, wind_speed, water_temp]
    env_data = torch.tensor([
        [0.5, 0.3, 5.0, 25.0],   # Calm conditions
        [1.5, 0.8, 12.0, 22.0],  # Moderate conditions
        [2.5, 1.5, 18.0, 15.0],  # Dangerous conditions
    ], dtype=torch.float32)
    
    risk_multipliers = fusion_model.calculate_environmental_risk(env_data)
    print(f"Environmental risk multipliers: {risk_multipliers}")
    
    # Test prediction with context
    print("\nTesting prediction with environmental context...")
    # Use just one sample for context testing
    single_motion = motion_processed[:1]
    single_hr = hr_processed[:1]
    single_env = env_data[:1]
    
    prediction, probs = fusion_model.predict_with_context(single_motion, single_hr, single_env)
    print(f"Prediction with context: {prediction}")
    print(f"Probabilities with context: {probs}")
    
    # Test without context
    prediction_no_context, probs_no_context = fusion_model.predict_with_context(single_motion, single_hr)
    print(f"Prediction without context: {prediction_no_context}")
    print(f"Probabilities without context: {probs_no_context}")
    
    print("\n[PASS] Fusion model test completed successfully!")


def test_model_saving_loading():
    """Test model saving and loading"""
    print("\n=== Testing Model Saving and Loading ===")
    
    # Create a fusion model
    fusion_model = FusionClassifier()
    
    # Save the model
    fusion_model.save_model("test_fusion_model.pth")
    
    # Create a new model and load weights
    loaded_model = FusionClassifier()
    try:
        loaded_model.load_model("test_fusion_model.pth")
        print("[PASS] Model saving and loading test passed!")
    except Exception as e:
        print(f"[FAIL] Model saving and loading test failed: {e}")
    
    # Clean up test file
    import os
    if os.path.exists("test_fusion_model.pth"):
        os.remove("test_fusion_model.pth")


if __name__ == "__main__":
    test_fusion_model()
    test_model_saving_loading()