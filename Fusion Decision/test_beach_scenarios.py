import torch
import numpy as np
from fusion_preproces import FusionDataProcessor
from fusion_train import FusionAITrainer

def test_beach_scenarios():
    """Test the beach-specific drowning detection system"""
    print("=== Beach Drowning Detection Test ===")
    
    # Initialize components
    preprocessor = FusionDataProcessor()
    trainer = FusionAITrainer()
    
    # Test different beach scenarios
    scenarios = {
        'normal_swimming': {
            'description': 'Normal swimming in calm water',
            'motion_variance': 0.8,
            'hr_mean': 75,
            'env_conditions': [0.3, 0.2, 5.0, 25.0]  # Low waves, weak current, light wind, warm water
        },
        'wave_riding': {
            'description': 'Riding waves (normal activity)',
            'motion_variance': 2.5,
            'hr_mean': 110,
            'env_conditions': [1.5, 0.8, 12.0, 22.0]  # Medium waves, moderate current, moderate wind
        },
        'drowning_panic': {
            'description': 'Drowning panic in rough water',
            'motion_variance': 3.5,
            'hr_mean': 150,
            'env_conditions': [2.0, 1.2, 15.0, 18.0]  # High waves, strong current, high wind, cold water
        },
        'rip_current_struggle': {
            'description': 'Struggling in rip current',
            'motion_variance': 3.0,
            'hr_mean': 140,
            'env_conditions': [1.0, 1.5, 8.0, 20.0]  # Medium waves, very strong current
        },
        'cold_water_shock': {
            'description': 'Cold water shock response',
            'motion_variance': 2.8,
            'hr_mean': 160,
            'env_conditions': [0.5, 0.3, 3.0, 12.0]  # Low waves, weak current, cold water
        }
    }
    
    print("\nTesting different beach scenarios:")
    print("=" * 50)
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        print(f"Description: {scenario_data['description']}")
        
        # Generate synthetic data for this scenario
        motion_data = generate_scenario_motion(scenario_data['motion_variance'])
        hr_data = generate_scenario_hr(scenario_data['hr_mean'])
        env_data = np.array([scenario_data['env_conditions']])
        
        # Classify activity (flatten data for classification)
        motion_flat = motion_data.reshape(-1)  # Flatten to 1D for variance calculation
        hr_flat = hr_data.reshape(-1)          # Flatten to 1D for mean calculation
        activity, confidence = preprocessor.beach_activity_classifier.classify(motion_flat, hr_flat)
        
        # Get prediction (if models are trained)
        try:
            prediction, probs = trainer.predict_beach_drowning(
                motion_data, 
                hr_data, 
                env_data
            )
            
            risk_level = trainer.class_names[prediction.item()]
            print(f"Activity Classification: {activity} (confidence: {confidence:.2f})")
            print(f"Risk Level: {risk_level}")
            print(f"Environmental Risk Multiplier: {trainer.calculate_environmental_risk(torch.tensor(env_data, dtype=torch.float32)).item():.2f}")
            
        except FileNotFoundError:
            print("Models not found - run training first")
            print(f"Activity Classification: {activity} (confidence: {confidence:.2f})")
        
        print("-" * 30)

def generate_scenario_motion(variance):
    """Generate motion data with specific variance"""
    np.random.seed(42)
    # Generate 6-channel motion data (3 accel + 3 gyro) with 100 timesteps
    motion = np.random.normal(0, np.sqrt(variance), (100, 6))
    # Reshape to (1, 6, 100) for batch processing
    motion = motion.T.reshape(1, 6, 100)
    return motion

def generate_scenario_hr(mean):
    """Generate heart rate data with specific mean"""
    np.random.seed(42)
    # Generate heart rate data with 100 timesteps
    hr = np.random.normal(mean, 10, 100)
    # Reshape to (1, 100, 1) for batch processing
    hr = hr.reshape(1, 100, 1)
    return hr

def test_environmental_risk_calculation():
    """Test environmental risk calculation"""
    print("\n=== Environmental Risk Calculation Test ===")
    
    trainer = FusionAITrainer()
    
    # Test different environmental conditions
    test_conditions = [
        [0.5, 0.3, 5.0, 25.0],   # Calm conditions
        [1.5, 0.8, 12.0, 22.0],  # Moderate conditions
        [2.5, 1.5, 18.0, 15.0],  # Dangerous conditions
        [0.2, 0.1, 2.0, 30.0],   # Very calm conditions
        [3.0, 2.0, 25.0, 10.0],  # Extreme conditions
    ]
    
    condition_names = ['Calm', 'Moderate', 'Dangerous', 'Very Calm', 'Extreme']
    
    for i, (condition, name) in enumerate(zip(test_conditions, condition_names)):
        env_data = torch.tensor([condition], dtype=torch.float32)
        risk = trainer.calculate_environmental_risk(env_data)
        
        print(f"{name} Conditions:")
        print(f"  Wave Height: {condition[0]}m")
        print(f"  Current Strength: {condition[1]} m/s")
        print(f"  Wind Speed: {condition[2]} m/s")
        print(f"  Water Temperature: {condition[3]}°C")
        print(f"  Risk Multiplier: {risk.item():.2f}")
        print()

def test_activity_classification():
    """Test beach activity classification"""
    print("\n=== Beach Activity Classification Test ===")
    
    preprocessor = FusionDataProcessor()
    
    # Test different activity patterns
    activities = {
        'swimming': {'motion_variance': 1.0, 'hr_mean': 80},
        'wave_riding': {'motion_variance': 2.5, 'hr_mean': 110},
        'floating': {'motion_variance': 0.3, 'hr_mean': 65},
        'playing': {'motion_variance': 1.8, 'hr_mean': 95},
        'drowning': {'motion_variance': 3.2, 'hr_mean': 140},
    }
    
    for activity_name, params in activities.items():
        motion_data = generate_scenario_motion(params['motion_variance'])
        hr_data = generate_scenario_hr(params['hr_mean'])
        
        classified_activity, confidence = preprocessor.beach_activity_classifier.classify(motion_data, hr_data)
        
        print(f"True Activity: {activity_name}")
        print(f"Classified As: {classified_activity}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Motion Variance: {params['motion_variance']:.1f}")
        print(f"Heart Rate Mean: {params['hr_mean']:.0f}")
        print("-" * 20)

if __name__ == "__main__":
    test_beach_scenarios()
    test_environmental_risk_calculation()
    test_activity_classification()
