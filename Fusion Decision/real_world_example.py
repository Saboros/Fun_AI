import numpy as np
import torch
from data_adapter import FlexibleDataAdapter
from fusion_train import FusionAITrainer
from fusion_preproces import FusionDataProcessor

def real_world_drowning_detection():
    """Example of using the fusion AI with real-world data formats"""
    print("=== Real-World Drowning Detection Example ===")
    
    # Initialize components
    adapter = FlexibleDataAdapter()
    trainer = FusionAITrainer()
    preprocessor = FusionDataProcessor()
    
    # Example 1: Smartphone sensor data (3-axis accelerometer only)
    print("\n📱 Example 1: Smartphone Data (3-axis accelerometer)")
    print("-" * 50)
    
    # Simulate smartphone accelerometer data (50Hz, 30 seconds = 1500 samples)
    smartphone_accel = np.random.normal(0, 1, (1500, 3))  # (timesteps, 3_axes)
    
    # Single heart rate reading from smartwatch
    smartwatch_hr = 95.0
    
    # Environmental data from weather app
    weather_data = {
        'wave_height': 0.8,
        'current_strength': 0.5,
        'wind_speed': 8.0,
        'water_temp': 24.0
    }
    
    # Adapt data to AI format
    motion_adapted = adapter.adapt_motion_data(smartphone_accel, 'smartphone')
    hr_adapted = adapter.adapt_heart_rate_data(smartwatch_hr, 'smartwatch')
    env_adapted = adapter.adapt_environmental_data(weather_data, 'weather_app')
    
    # Get prediction
    try:
        prediction, probs = trainer.predict_beach_drowning(motion_adapted, hr_adapted, env_adapted)
        risk_level = trainer.class_names[prediction.item()]
        print(f"Risk Level: {risk_level}")
        print(f"Environmental Risk: {trainer.calculate_environmental_risk(env_adapted).item():.2f}x")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    # Example 2: Professional IMU sensor (9-axis, high frequency)
    print("\n🏊 Example 2: Professional IMU Sensor (9-axis)")
    print("-" * 50)
    
    # Simulate professional IMU data (200Hz, 10 seconds = 2000 samples)
    professional_imu = np.random.normal(0, 1, (2000, 9))  # (timesteps, 9_axes)
    
    # Heart rate sequence from chest strap
    chest_strap_hr = np.random.normal(120, 15, 2000)  # High frequency HR data
    
    # Environmental data from beach sensors
    beach_sensors = np.array([1.5, 1.2, 15.0, 18.0])  # [wave, current, wind, temp]
    
    # Adapt data
    motion_adapted = adapter.adapt_motion_data(professional_imu, 'professional_imu')
    hr_adapted = adapter.adapt_heart_rate_data(chest_strap_hr, 'chest_strap')
    env_adapted = adapter.adapt_environmental_data(beach_sensors, 'beach_sensors')
    
    # Get prediction
    try:
        prediction, probs = trainer.predict_beach_drowning(motion_adapted, hr_adapted, env_adapted)
        risk_level = trainer.class_names[prediction.item()]
        print(f"Risk Level: {risk_level}")
        print(f"Environmental Risk: {trainer.calculate_environmental_risk(env_adapted).item():.2f}x")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    # Example 3: Mixed data sources (incomplete data)
    print("\n🌊 Example 3: Mixed Data Sources (Incomplete)")
    print("-" * 50)
    
    # Only accelerometer data available
    basic_accel = np.random.normal(0, 2, (100, 3))  # High variance (panic motion)
    
    # No heart rate data available
    no_hr_data = None
    
    # Only wave height available
    partial_env = {'wave_height': 2.5}  # Missing other environmental data
    
    # Adapt data (handles missing data gracefully)
    motion_adapted = adapter.adapt_motion_data(basic_accel, 'basic_accel')
    hr_adapted = adapter.adapt_heart_rate_data(no_hr_data, 'missing')
    env_adapted = adapter.adapt_environmental_data(partial_env, 'partial')
    
    # Get prediction
    try:
        prediction, probs = trainer.predict_beach_drowning(motion_adapted, hr_adapted, env_adapted)
        risk_level = trainer.class_names[prediction.item()]
        print(f"Risk Level: {risk_level}")
        print(f"Environmental Risk: {trainer.calculate_environmental_risk(env_adapted).item():.2f}x")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    # Example 4: CSV data from file (simulated)
    print("\n📊 Example 4: CSV Data from File")
    print("-" * 50)
    
    # Simulate CSV data (different sampling rates)
    csv_motion = np.random.normal(0, 1.5, (75, 6))  # 75 timesteps, 6 channels
    csv_hr = np.random.normal(140, 20, 75)  # High heart rate (panic)
    csv_env = np.array([2.0, 1.5, 20.0, 15.0])  # Dangerous conditions
    
    # Adapt CSV data
    motion_adapted = adapter.adapt_motion_data(csv_motion, 'csv_motion')
    hr_adapted = adapter.adapt_heart_rate_data(csv_hr, 'csv_hr')
    env_adapted = adapter.adapt_environmental_data(csv_env, 'csv_env')
    
    # Get prediction
    try:
        prediction, probs = trainer.predict_beach_drowning(motion_adapted, hr_adapted, env_adapted)
        risk_level = trainer.class_names[prediction.item()]
        print(f"Risk Level: {risk_level}")
        print(f"Environmental Risk: {trainer.calculate_environmental_risk(env_adapted).item():.2f}x")
    except Exception as e:
        print(f"Prediction failed: {e}")

def demonstrate_flexibility():
    """Demonstrate the flexibility of the data adapter"""
    print("\n=== Data Adapter Flexibility Demo ===")
    
    adapter = FlexibleDataAdapter()
    
    # Show different input formats that work
    print("\n✅ Supported Input Formats:")
    print("📱 Motion Data:")
    print("   - 3-axis accelerometer (50Hz, 30s)")
    print("   - 9-axis IMU (200Hz, 10s)")
    print("   - 6-axis motion (100Hz, 5s)")
    print("   - CSV files, JSON, numpy arrays")
    
    print("\n💓 Heart Rate Data:")
    print("   - Single value (85 BPM)")
    print("   - Time series (1000 samples)")
    print("   - Smartwatch data")
    print("   - Chest strap data")
    print("   - Missing data (handled gracefully)")
    
    print("\n🌊 Environmental Data:")
    print("   - Dictionary format")
    print("   - Array format")
    print("   - Partial data (missing values)")
    print("   - Weather API data")
    print("   - Beach sensor data")
    
    print("\n🔄 Automatic Adaptations:")
    print("   - Resampling to 100 timesteps")
    print("   - Channel padding/truncation")
    print("   - Missing data handling")
    print("   - Format conversion")
    print("   - Shape normalization")

def practical_usage_tips():
    """Provide practical tips for real-world usage"""
    print("\n=== Practical Usage Tips ===")
    
    print("\n🎯 For Different Devices:")
    print("📱 Smartphone: Use 3-axis accelerometer + single HR")
    print("⌚ Smartwatch: Use 6-axis motion + continuous HR")
    print("🏊 Professional: Use 9-axis IMU + chest strap HR")
    print("📊 Research: Use CSV/JSON files from sensors")
    
    print("\n⚡ Performance Considerations:")
    print("   - Higher sampling rates = better accuracy")
    print("   - Longer sequences = more context")
    print("   - More sensors = richer data")
    print("   - Real-time processing possible")
    
    print("\n🔧 Integration Tips:")
    print("   - Use the adapter for any data format")
    print("   - Handle missing data gracefully")
    print("   - Scale to multiple users")
    print("   - Add data validation")
    print("   - Implement error handling")

if __name__ == "__main__":
    real_world_drowning_detection()
    demonstrate_flexibility()
    practical_usage_tips()
    
    print("\n🎉 Your fusion AI is now ready for real-world data!")
    print("No need to worry about exact dataset formats anymore!")
