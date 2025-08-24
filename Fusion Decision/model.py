import torch
import torch.nn as nn
from heartmodel import HeartRateAnomalyDetector
from motion_classifier import MotionClassifier


class FusionClassifier(nn.Module):
    def __init__(self, motion_input_dim=3, heart_input_dim=1, fusion_output_dim=4):
        super(FusionClassifier, self).__init__()
        
        # Initialize individual models
        self.motion_classifier = MotionClassifier()
        self.heart_rate_detector = HeartRateAnomalyDetector()
        
        # Freeze individual models during fusion training
        for param in self.motion_classifier.parameters():
            param.requires_grad = False
            
        for param in self.heart_rate_detector.parameters():
            param.requires_grad = False

        # Fusion Layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(4, 64),  # 3 motion classes + 1 heart rate
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, fusion_output_dim)
        )

        # Decision Logic
        self.decision_threshold = {
            'emergency': 0.8,
            'warning': 0.6,
            'alert': 0.4,
            'normal': 0.2
        }
    
    def forward(self, motion_data, heart_rate_data):
        """
        Forward pass through the fusion classifier
        
        Args:
            motion_data: Tensor of shape (batch_size, channels, sequence_length)
            heart_rate_data: Tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            final_decision: Tensor of shape (batch_size, fusion_output_dim)
        """
        # Get predictions from individual models
        with torch.no_grad():  # Ensure individual models are not updated
            motion_pred = self.motion_classifier(motion_data)
            hr_pred = self.heart_rate_detector(heart_rate_data)

        # Combine predictions
        motion_probs = torch.softmax(motion_pred, dim=1)
        hr_probs = torch.sigmoid(hr_pred)
        
        # Concatenate features
        combined = torch.cat([motion_probs, hr_probs], dim=1)

        # Final decision through fusion layer
        final_decision = self.fusion_layer(combined)

        return final_decision
    
    def predict_with_context(self, motion_data, heart_rate_data, environmental_data=None):
        """
        Make prediction with environmental context
        
        Args:
            motion_data: Tensor of shape (batch_size, channels, sequence_length)
            heart_rate_data: Tensor of shape (batch_size, sequence_length, features)
            environmental_data: Optional tensor of shape (batch_size, features)
            
        Returns:
            prediction: Predicted class
            probabilities: Class probabilities
        """
        # Get base prediction
        with torch.no_grad():
            logits = self.forward(motion_data, heart_rate_data)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Apply environmental context if available
        if environmental_data is not None:
            environmental_risk = self.calculate_environmental_risk(environmental_data)
            # Adjust probabilities based on environmental risk
            adjusted_probs = probabilities * environmental_risk.unsqueeze(1)
            prediction = torch.argmax(adjusted_probs, dim=1)
            
        return prediction, probabilities
    
    def calculate_environmental_risk(self, env_data):
        """
        Calculate environmental risk multiplier
        
        Args:
            env_data: Tensor of shape (batch_size, features)
                Features: [wave_height, current_strength, wind_speed, water_temp]
                
        Returns:
            risk_multiplier: Tensor of shape (batch_size,)
        """
        # Extract environmental factors
        wave_height = env_data[:, 0]
        current_strength = env_data[:, 1]
        wind_speed = env_data[:, 2]
        water_temp = env_data[:, 3]
        
        # Calculate risk multipliers (normalized to [1.0, 3.0])
        wave_risk = torch.clamp(wave_height / 1.0, 1.0, 3.0)
        current_risk = torch.clamp(current_strength / 0.5, 1.0, 2.5)
        wind_risk = torch.clamp(wind_speed / 10.0, 1.0, 2.0)
        temp_risk = torch.where(water_temp < 20, 1.5, 1.0)
        
        # Combine risk factors
        total_risk = wave_risk * current_risk * wind_risk * temp_risk
        
        return total_risk
    
    def save_model(self, path):
        """Save the fusion model"""
        torch.save(self.state_dict(), path)
        print(f"Fusion model saved to {path}")
        
    def load_model(self, path):
        """Load the fusion model"""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
        print(f"Fusion model loaded from {path}")