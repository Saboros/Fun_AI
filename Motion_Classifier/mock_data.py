import torch
import torch.nn as nn
import torch.optim as optim
from motion_classifier import MotionClassifier
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

class MotionDataHandler:
    def __init__(self, sequence_length: int = 100, n_features: int = 6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.classes =  ['normal', 'panic', 'immobile']
        self.range_thresholds = {
            'max_distance': 100, #In meters
            'signal_strengh_threshold': -70, #IN dBm
            'acceleration_threshold': 16 #IN MS
        }
    
    def check_range(self, signal_strength: float, distance: float = None) -> bool:
        """Check if device is within acceptable range"""
        if distance and distance > self.range_thresholds['max_distance']:
            return False
        if signal_strength < self.range_thresholds['signal_strengh_threshold']:
            return False
        return True

    def generate_mock_data(self, n_samples: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic motion data for testing"""
        normal_data = np.sin(np.linspace(0, 10, self.sequence_length)) + \
                       np.random.normal(0, 0.1, (n_samples//3, self.n_features, self.sequence_length))
        
        panic_data = np.sin(np.linspace(0, 20, self.sequence_length)) + \
                       np.random.normal(0, 0.3, (n_samples//3, self.n_features, self.sequence_length))
        
        immobile_data = np.random.normal(0, 0.5, (n_samples//3, self.n_features, self.sequence_length))
       

        X = np.concatenate([normal_data, panic_data, immobile_data], axis = 0)
        y = np.concatenate([
            np.zeros(n_samples//3),
            np.ones(n_samples//3),
            np.ones(n_samples//3) * 2
        ])

        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def generate_labels_from_motion(self, motion_data: np.ndarray) -> np.ndarray:
        """Generate labels based on motion patterns"""
        labels = []
        
        for i in range(len(motion_data)):
            # Calculate motion variance for each sample
            motion_variance = np.var(motion_data[i])
            motion_mean = np.mean(np.abs(motion_data[i]))
            
            # Classify based on motion characteristics
            if motion_variance < 0.1 and motion_mean < 0.05:
                # Low variance and low mean = immobile
                labels.append(2)  # immobile
            elif motion_variance > 0.5 or motion_mean > 0.3:
                # High variance or high mean = panic
                labels.append(1)  # panic
            else:
                # Medium motion = normal
                labels.append(0)  # normal
        
        return np.array(labels)
    
    def load_csv_data(self, file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load CSV Data - Fixed to handle S1-ADL1 format"""
        import pandas as pd
        
        print(f"Loading data from: {file_path}")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we have motion data columns
        motion_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        if all(col in df.columns for col in motion_columns):
            print("✅ Found motion data columns")
            features = motion_columns
        elif 'acc_x' in df.columns and 'acc_y' in df.columns:
            print("✅ Found alternative motion data columns")
            features = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        else:
            # Use first 6 columns as motion data
            print("⚠️ Using first 6 columns as motion data")
            features = df.columns[:6].tolist()
        
        print(f"Using features: {features}")
        
        # Extract motion data
        motion_data = df[features].values
        print(f"Motion data shape: {motion_data.shape}")
        
        # Check if we have labels
        if 'label' in df.columns:
            print("✅ Found label column")
            labels = df['label'].values
        elif 'activity_label' in df.columns:
            print("✅ Found activity_label column")
            # Convert string labels to numeric
            label_mapping = {'normal': 0, 'panic': 1, 'immobile': 2, 'drowning': 1}
            labels = [label_mapping.get(label, 0) for label in df['activity_label']]
        else:
            print("⚠️ No labels found - generating labels from motion patterns")
            labels = self.generate_labels_from_motion(motion_data)
        
        # Create sequences
        sequences = []
        sequence_labels = []
        
        # Segment data into sequences
        n_sequences = len(motion_data) // self.sequence_length
        
        print(f"Creating {n_sequences} sequences of length {self.sequence_length}")
        
        for i in range(n_sequences):
            start_idx = i * self.sequence_length
            end_idx = start_idx + self.sequence_length
            
            # Extract sequence
            sequence = motion_data[start_idx:end_idx]
            
            # Transpose to (features, timesteps) format
            sequence = sequence.T
            
            # Get label for this sequence (use majority label)
            sequence_label_data = labels[start_idx:end_idx]
            sequence_label = int(np.round(np.mean(sequence_label_data)))
            
            sequences.append(sequence)
            sequence_labels.append(sequence_label)
        
        # Convert to tensors
        X = torch.FloatTensor(sequences)
        y = torch.LongTensor(sequence_labels)
        
        print(f"Final data shape: X={X.shape}, y={y.shape}")
        print(f"Unique labels: {set(sequence_labels)}")
        print(f"Label distribution: {np.bincount(sequence_labels)}")
        
        return X, y
    
    def test_range_detection():
        """Test the range detection functionality"""
        data_handler = MotionDataHandler()

        X, y, range_info, range_violations = data_handler.generate_mock_data(n_samples=1000)

        for i in range(5):
            signal_strength = range_info[i][0].item()
            distance = range_info[i][1].item()
            is_in_range = data_handler.check_range(signal_strength, distance)

            print(f"\nSample {i+1}:")
            print(f"Signal Strength: {signal_strength:.1f} dBm")
            print(f"Distance: {distance:.1f}m")
            print(f"In Range: {is_in_range}")
            if not is_in_range:
                print("WARNING: Device out of range!")

    def preprocess_data(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data for training"""
        # Normalize features
        X = (X -X.mean(dim = 2, keepdim = True)) / (X.std(dim = 2, keepdim = True) + 1e-7)
        
        if y is not None:
            indices = torch.randperm(len(X))
            return X[indices], y[indices]
        else:
            return X, None

if __name__ == "__main__":
    data_handler = MotionDataHandler()

    #FOR MOCK
    X_mock, y_mock = data_handler.generate_mock_data(n_samples = 300)
    X_mock, y_mock = data_handler.preprocess_data(X_mock, y_mock)

    print(f"Mock data shape: {X_mock.shape}, Labels shape: {y_mock.shape}")

    #FOR CSV
    example_data = {
        'timestamp': range(1000),
        'acc_x': np.random.randn(1000),
        'acc_y': np.random.randn(1000),
        'acc_z': np.random.randn(1000),
        'gyro_x': np.random.randn(1000),
        'gyro_y': np.random.randn(1000),
        'gyro_z': np.random.randn(1000),
        'label': np.random.choice([0, 1, 2], 1000)
    }

    df = pd.DataFrame(example_data)
    df.to_csv('example_motion_data.csv', index=False)
