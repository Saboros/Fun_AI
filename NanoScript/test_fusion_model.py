import torch
import numpy as np
from model import MultiModalFusionClassifier
from fusion_preproces import FusionDataProcessor
from hrpreprocess import denormalize_spo2
import warnings
warnings.filterwarnings("ignore")


class FusionModelTester:
    """Test the trained fusion model with various scenarios"""
    
    def __init__(self, model_path='Models/fusion_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.class_names = ['Emergency', 'Warning', 'Alert', 'Normal']
        self.preprocessor = FusionDataProcessor()
        
        # Load model
        print("Loading fusion model...")
        self.model = MultiModalFusionClassifier().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✓ Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"✗ Model not found at {model_path}")
            print("Please train the model first by running: python fusion_train.py")
            exit(1)
    
    def create_test_scenario(self, scenario_type):
        """Create specific test scenarios"""
        if scenario_type == 'drowning':
            print("\n📍 Scenario: Drowning (Emergency)")
            # Panic motion + High HR + Low SpO2
            motion = np.random.normal(0, 2.5, (1, 6, 100)).astype(np.float32)
            # Add erratic spikes
            motion[:, :, ::10] += np.random.uniform(3, 6, (1, 6, 10))
            hr = np.random.normal(150, 10, (1, 100)).astype(np.float32)
            spo2 = 78.0  # Low SpO2
            
        elif scenario_type == 'unconscious':
            print("\n📍 Scenario: Unconscious (Warning)")
            # Immobile motion + Low SpO2
            motion = np.random.normal(0, 0.1, (1, 6, 100)).astype(np.float32)
            hr = np.random.normal(55, 3, (1, 100)).astype(np.float32)
            spo2 = 85.0  # Low SpO2
            
        elif scenario_type == 'exhaustion':
            print("\n📍 Scenario: Exhaustion (Alert)")
            # Normal motion + High HR + Normal SpO2
            motion = np.random.normal(0, 0.6, (1, 6, 100)).astype(np.float32)
            hr = np.random.normal(130, 8, (1, 100)).astype(np.float32)
            spo2 = 96.0  # Normal SpO2
            
        elif scenario_type == 'swimming':
            print("\n📍 Scenario: Normal Swimming (Normal)")
            # Normal motion + Normal HR + Normal SpO2
            motion = np.random.normal(0, 0.8, (1, 6, 100)).astype(np.float32)
            hr = np.random.normal(85, 5, (1, 100)).astype(np.float32)
            spo2 = 98.0  # Normal SpO2
            
        elif scenario_type == 'panic':
            print("\n📍 Scenario: Panic/Struggling (Emergency)")
            # Very erratic motion + Very high HR + Dropping SpO2
            motion = np.random.normal(0, 3.0, (1, 6, 100)).astype(np.float32)
            motion[:, :, ::5] += np.random.uniform(4, 8, (1, 6, 20))
            hr = np.random.normal(170, 15, (1, 100)).astype(np.float32)
            spo2 = 82.0  # Low SpO2
            
        else:  # resting
            print("\n📍 Scenario: Resting/Floating (Normal)")
            # Low motion + Low HR + Normal SpO2
            motion = np.random.normal(0, 0.3, (1, 6, 100)).astype(np.float32)
            hr = np.random.normal(68, 4, (1, 100)).astype(np.float32)
            spo2 = 99.0  # Normal SpO2
        
        return motion, hr, spo2
    
    def predict_with_details(self, motion_data, hr_data, spo2_value):
        """Make prediction and show detailed analysis"""
        # Preprocess data
        motion_processed = self.preprocessor.preprocess_motion_data(motion_data)
        hr_processed = self.preprocessor.preprocess_hr_data(hr_data)
        
        # Move to device
        motion_tensor = motion_processed.to(self.device)
        hr_tensor = hr_processed.to(self.device)
        
        # Get prediction with interpretability
        with torch.no_grad():
            outputs = self.model(motion_tensor, hr_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
            # Get interpretability info (pass the tensors)
            interp_info = self.model.get_interpretability_info(motion_tensor, hr_tensor)
        
        # Display results
        print(f"\n📊 Input Vitals:")
        print(f"   Heart Rate: {np.mean(hr_data):.1f} bpm (range: {np.min(hr_data):.1f}-{np.max(hr_data):.1f})")
        print(f"   SpO2: {spo2_value:.1f}%")
        print(f"   Motion Variance: {np.var(motion_data):.3f}")
        
        print(f"\n🎯 Prediction: {self.class_names[prediction]}")
        print(f"   Confidence: {probabilities[0][prediction].item()*100:.1f}%")
        
        print(f"\n📈 Class Probabilities:")
        for i, class_name in enumerate(self.class_names):
            prob = probabilities[0][i].item() * 100
            bar = '█' * int(prob / 2)
            print(f"   {class_name:12s}: {prob:5.1f}% {bar}")
        
        # Show attention weights
        if 'motion_to_physio_attention' in interp_info:
            motion_attn = interp_info['motion_to_physio_attention'][0].mean().item()
            physio_attn = interp_info['physio_to_motion_attention'][0].mean().item()
            print(f"\n🔍 Cross-Modal Attention:")
            print(f"   Motion → Physio: {motion_attn:.3f}")
            print(f"   Physio → Motion: {physio_attn:.3f}")
        
        # Emergency detection
        if 'emergency_gate' in interp_info:
            emergency_score = interp_info['emergency_gate'][0].item()
            print(f"\n⚠️  Emergency Score: {emergency_score:.3f}")
            if emergency_score > 0.7:
                print("   ⚠️  HIGH RISK - Immediate attention required!")
            elif emergency_score > 0.4:
                print("   ⚠️  MODERATE RISK - Monitor closely")
            else:
                print("   ✓ Low risk")
        
        return prediction, probabilities
    
    def test_batch_scenarios(self):
        """Test multiple scenarios"""
        print("=" * 60)
        print("FUSION MODEL COMPREHENSIVE TEST")
        print("=" * 60)
        
        scenarios = ['drowning', 'unconscious', 'exhaustion', 'swimming', 'panic', 'resting']
        
        correct_predictions = 0
        expected_classes = {
            'drowning': 0,      # Emergency
            'unconscious': 1,   # Warning
            'exhaustion': 2,    # Alert
            'swimming': 3,      # Normal
            'panic': 0,         # Emergency
            'resting': 3        # Normal
        }
        
        for scenario in scenarios:
            motion, hr, spo2 = self.create_test_scenario(scenario)
            prediction, probs = self.predict_with_details(motion, hr, spo2)
            
            expected = expected_classes[scenario]
            if prediction == expected:
                correct_predictions += 1
                print(f"✓ Correct prediction!")
            else:
                print(f"✗ Expected: {self.class_names[expected]}")
            
            print("-" * 60)
        
        accuracy = (correct_predictions / len(scenarios)) * 100
        print(f"\n🎯 Scenario Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(scenarios)})")
    
    def test_with_generated_data(self, num_samples=100):
        """Test with generated test dataset"""
        print("\n" + "=" * 60)
        print("TESTING WITH GENERATED DATASET")
        print("=" * 60)
        
        # Generate test data
        print("\nGenerating test data...")
        motion_data, hr_data, spo2_data, labels = self.preprocessor.create_fusion_dataset(
            num_samples=num_samples, seed=123
        )
        
        # Preprocess
        motion_processed = self.preprocessor.preprocess_motion_data(motion_data)
        hr_processed = self.preprocessor.preprocess_hr_data(hr_data)
        
        # Move to device
        motion_tensor = motion_processed.to(self.device)
        hr_tensor = hr_processed.to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Predict
        print("Running predictions...")
        with torch.no_grad():
            outputs = self.model(motion_tensor, hr_tensor)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy
            correct = (predictions == labels_tensor).sum().item()
            accuracy = correct / num_samples * 100
        
        # Per-class accuracy
        print(f"\n📊 Overall Accuracy: {accuracy:.2f}%")
        print(f"\n📈 Per-Class Performance:")
        
        for i, class_name in enumerate(self.class_names):
            class_mask = labels_tensor == i
            if class_mask.sum() > 0:
                class_correct = ((predictions == labels_tensor) & class_mask).sum().item()
                class_total = class_mask.sum().item()
                class_acc = class_correct / class_total * 100
                print(f"   {class_name:12s}: {class_acc:5.1f}% ({class_correct}/{class_total})")
        
        # Confusion matrix
        print(f"\n🔢 Confusion Matrix:")
        print("   " + "  ".join([f"{name[:4]:>4s}" for name in self.class_names]))
        for i in range(len(self.class_names)):
            row = []
            for j in range(len(self.class_names)):
                count = ((labels_tensor == i) & (predictions == j)).sum().item()
                row.append(f"{count:>4d}")
            print(f"   {' '.join(row)}  <- {self.class_names[i]}")
    
    def test_custom_input(self):
        """Test with custom user input"""
        print("\n" + "=" * 60)
        print("CUSTOM INPUT TEST")
        print("=" * 60)
        
        print("\nEnter physiological parameters:")
        try:
            hr_input = float(input("Heart Rate (bpm, e.g., 75): "))
            spo2_input = float(input("SpO2 (%, e.g., 98): "))
            motion_var = float(input("Motion intensity (0-5, e.g., 0.5 for calm, 3.0 for panic): "))
            
            # Generate data based on input
            motion = np.random.normal(0, motion_var, (1, 6, 100)).astype(np.float32)
            hr = np.full((1, 100), hr_input, dtype=np.float32)
            
            print(f"\n📍 Scenario: Custom Input")
            prediction, probs = self.predict_with_details(motion, hr, spo2_input)
            
        except (ValueError, KeyboardInterrupt):
            print("\nSkipping custom input test.")


def main():
    """Main test function"""
    tester = FusionModelTester()
    
    # Test scenarios
    tester.test_batch_scenarios()
    
    # Test with generated data
    tester.test_with_generated_data(num_samples=200)
    
    # Optional: Test with custom input
    print("\n" + "=" * 60)
    custom_test = input("\nWould you like to test with custom input? (y/n): ")
    if custom_test.lower() == 'y':
        tester.test_custom_input()
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
