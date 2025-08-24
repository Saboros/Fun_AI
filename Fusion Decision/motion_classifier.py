import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque
import os
import warnings
warnings.filterwarnings("ignore")


class MotionClassifier(nn.Module):
    """
    1D-CNN for motion classification
    Input: (batch_size, 6, 100) - 6 channels (3 accel + 3 gyro) x 100 timesteps
    Output: 3 classes (normal, panic, immobile)
    """
    def __init__(self, input_channels=6, sequence_length=100, num_classes=3):
        super(MotionClassifier, self).__init__()

        # Neural Network Layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)
        )

        self.flatten_size = 128 * (sequence_length // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
    
    def save_model(self, path="Models/motion_classifier.pth"):
        """Save the model checkpoint"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path="Models/motion_classifier.pth"):
        """Load the model checkpoint"""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    model = MotionClassifier()
    model.save_model('motion_classifier.pth')

    loaded_model = MotionClassifier()
    loaded_model.load_model('motion_classifier.pth')
