import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings("ignore")


#MOTION CLASSIFIER(1D-CNN)

class MotionClassifier(nn.Module):
    """
    1D-CNN for motion classification
    Input: (batch_size, 6, 100) - 6 channels (3 accel + 3 gyro) x 100 timesteps
    Output: 3 classes (normal, panic, immobile)

    """
    def __init__(self, input_channels = 6, sequence_length = 100, num_classes = 3):
        super(MotionClassifier, self).__init__()

        #Neural Network Layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size = 5, padding = 2 ),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)
        )

        self.flatten_size =  128 * (sequence_length // 8)

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

path = "D:/_Python Projects/Fun_AI/Models"

def save_model(model, path):
        torch.save(model.state_dict(), path)

def load_model(model, path):
        model.load_state_dict(torch.load(path))
        model.eval()
    
if __name__ == "__main__":
    model = MotionClassifier()
    save_model(model, 'motion_classifier.pth')

    loaded_model = MotionClassifier()
    load_model(loaded_model, 'motion_classifier.pth')
