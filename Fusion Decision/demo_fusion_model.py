"""
Demo script for the refactored fusion model
This script demonstrates how to use the refactored fusion model for drowning detection
"""

import torch
import numpy as np
from model import FusionClassifier
from fusion_preproces import FusionDataProcessor
from data_adapter import FlexibleDataAdapter
import warnings
warnings.filterwarnings("ignore")


def demo_basic_usage():
    """Demonstrate basic usage of the fusion model"""
    print("=== Basic Fusion Model Usage Demo ===")
    
    # Initialize components
    fusion_model = FusionClassifier()
    preprocessor = FusionDataProcessor()
    data_adapter = FlexibleDataAdapter()
    
    # Generate sample data
    print("\n1. Generating sample data...")
    motion_data, hr_data, labels = preprocessor.create_fusion_dataset(num_samples=50)
    
    # Preprocess data
    print("2. Preprocessing data...")
    motion_processed = preprocessor.preprocess_motion_data(motion_data)
    hr_processed = preprocessor.preprocess_hr_data(hr_data)
    
    print(f"   Motion data shape: {motion_processed.shape}")
    print(f"   HR data shape: {hr_processed.shape}")
    
    # Make predictions
    print("\n3. Making predictions...")
    with torch.no_grad():
        logits = fusion_model(motion_processed, hr_processed)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:5].tolist()}")
    print(f"   Sample probabilities: {probabilities[:2].tolist()}")
    
    # Show class names
    class_names = ['Emergency', 'Warning', 'Alert', 'Normal']
    print(f"\n   Class names: {class_names}")


def demo_with_environmental_context():
    """Demonstrate usage with environmental context"""
    print("\n=== Fusion Model with Environmental Context Demo ===")
    
    # Initialize components
    fusion_model = FusionClassifier()
    preprocessor = FusionDataProcessor()
    
    # Generate sample data
    print("\n1. Generating sample data...")
    motion_data, hr_data, env_data, labels = preprocessor.create_beach_dataset(num_samples=10)
    
    # Preprocess data
    print("2. Preprocessing data...")
    motion_processed = preprocessor.preprocess_motion_data(motion_data)
    hr_processed = preprocessor.preprocess_hr_data(hr_data)
    env_processed = preprocessor.preprocess_environmental_data(env_data)
    
    print(f"   Motion data shape: {motion_processed.shape}")
    print(f"   HR data shape: {hr_processed.shape}")
    print(f"   Environmental data shape: {env_processed.shape}")
    
    # Make predictions with and without context
    print("\n3. Making predictions...")
    
    # Without context
    with torch.no_grad():
        logits = fusion_model(motion_processed, hr_processed)
        probs_no_context = torch.softmax(logits, dim=1)
        pred_no_context = torch.argmax(probs_no_context, dim=1)
    
    # With context
    pred_with_context, probs_with_context = fusion_model.predict_with_context(
        motion_processed, hr_processed, env_processed
    )
    
    print(f"   Predictions without context: {pred_no_context[:3].tolist()}")
    print(f"   Predictions with context: {pred_with_context[:3].tolist()}")
    print(f"   Sample environmental data: {env_data[:2].tolist()}")


def demo_data_adaptation():
    """Demonstrate the flexible data adapter"""
    print("\n=== Flexible Data Adapter Demo ===")
    
    # Initialize adapter
    data_adapter = FlexibleDataAdapter()
    
    # Create sample data in various formats
    print("\n1. Creating sample data in various formats...")
    sample_data = data_adapter.create_sample_data()
    
    # Adapt different data formats
    print("\n2. Adapting data to model format...")
    
    # Adapt 3-axis accelerometer data
    motion_3d = data_adapter.adapt_motion_data(sample_data['accel_3d'], '3d_accel')
    print(f"   3-axis accelerometer adapted shape: {motion_3d.shape}")
    
    # Adapt 9-axis IMU data
    motion_9d = data_adapter.adapt_motion_data(sample_data['imu_9d'], '9d_imu')
    print(f"   9-axis IMU adapted shape: {motion_9d.shape}")
    
    # Adapt single heart rate value
    hr_single = data_adapter.adapt_heart_rate_data(sample_data['hr_single'], 'single')
    print(f"   Single HR value adapted shape: {hr_single.shape}")
    
    # Adapt heart rate sequence
    hr_sequence = data_adapter.adapt_heart_rate_data(sample_data['hr_sequence'], 'sequence')
    print(f"   HR sequence adapted shape: {hr_sequence.shape}")
    
    # Adapt environmental data
    env_dict = data_adapter.adapt_environmental_data(sample_data['env_dict'], 'dict')
    print(f"   Environmental dict adapted shape: {env_dict.shape}")
    
    env_array = data_adapter.adapt_environmental_data(sample_data['env_array'], 'array')
    print(f"   Environmental array adapted shape: {env_array.shape}")


def main():
    """Run all demos"""
    print("Fusion Model Demo Suite")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_with_environmental_context()
        demo_data_adaptation()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()