import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# NEW: Beach activity classifier
class BeachActivityClassifier:
    def __init__(self):
        self.activity_patterns = {
            'swimming': {'motion_variance': (0.5, 1.5), 'hr_range': (70, 90)},
            'wave_riding': {'motion_variance': (1.5, 3.0), 'hr_range': (90, 120)},
            'floating': {'motion_variance': (0.1, 0.5), 'hr_range': (60, 75)},
            'playing': {'motion_variance': (1.0, 2.5), 'hr_range': (80, 110)},
            'drowning': {'motion_variance': (2.0, 4.0), 'hr_range': (120, 180)}
        }
    
    def classify(self, motion_data, hr_data):
        """Classify beach activity based on motion and heart rate patterns"""
        motion_variance = np.var(motion_data)
        hr_mean = np.mean(hr_data)
        
        best_match = 'unknown'
        best_score = 0
        
        for activity, patterns in self.activity_patterns.items():
            motion_match = self._check_range(motion_variance, patterns['motion_variance'])
            hr_match = self._check_range(hr_mean, patterns['hr_range'])
            score = motion_match * hr_match
            
            if score > best_score:
                best_score = score
                best_match = activity
        
        return best_match, best_score
    
    def _check_range(self, value, range_tuple):
        """Check if value falls within range and return similarity score"""
        min_val, max_val = range_tuple
        if min_val <= value <= max_val:
            return 1.0
        else:
            # Return similarity score based on distance from range
            distance = min(abs(value - min_val), abs(value - max_val))
            return max(0, 1 - distance / max_val)

class FusionDataProcessor:
    def __init__(self, motion_seq_len = 100, hr_seq_len = 100, fusion_seq_len = 100):
        self.motion_seq_len = motion_seq_len
        self.hr_seq_len = hr_seq_len
        self.fusion_seq_len = fusion_seq_len
        self.motion_scaler = StandardScaler()
        self.hr_scaler = StandardScaler()
        
        # NEW: Beach-specific components
        self.environmental_scaler = StandardScaler()
        self.beach_activity_classifier = BeachActivityClassifier()

    def create_fusion_dataset(self, num_samples = 1000, seed = 42):
        np.random.seed(seed)

        motion_data = []
        motion_labels = []

        for i in range(num_samples):
            if i < num_samples * 0.7:
                accel = np.random.normal(0,0.5, (self.motion_seq_len, 3))
                gyro = np.random.normal(0, 0.3, (self.motion_seq_len, 3))
                motion_labels.append(0)
            
            elif i < num_samples * 0.85:
                accel = np.random.normal(0, 2.0, (self.motion_seq_len, 3))
                gyro = np.random.normal(0, 1.5, (self.motion_seq_len, 3))
                motion_labels.append(1)
            
            else:
                accel = np.random.normal(0, 0.1, (self.motion_seq_len, 3))
                gyro = np.random.normal(0, 0.5, (self.motion_seq_len, 3))
                motion_labels.append(2)
            
            motion_data.append(np.concatenate([accel, gyro], axis = 1))
        
        hr_data = []
        hr_labels = []

        for i in range(num_samples):
            if motion_labels[i] == 0:
                hr_sequence = np.random.normal(75, 10, self.hr_seq_len)
                hr_labels.append(0)
            elif motion_labels[i] == 1:
                hr_sequence = np.random.normal(120, 20, self.hr_seq_len)
                hr_labels.append(1)
            else:
                hr_sequence = np.random.normal(60, 5, self.hr_seq_len)
                hr_labels.append(0)
        
            hr_data.append(hr_sequence)
        
        fusion_labels = []
        for i in range(num_samples):
            if motion_labels[i] == 1: #Panic Motion
                fusion_labels.append(0) # Emergency
            elif motion_labels[i] == 2: # Immobile
                fusion_labels.append(1) # Warning
            elif hr_labels[i] == 1: # High HR
                fusion_labels.append(2) # Emergency
            else:
                fusion_labels.append(3) # Normal
        return np.array(motion_data), np.array(hr_data), np.array(fusion_labels)

    def preprocess_motion_data(self, motion_data):
        # Handle different input shapes
        if len(motion_data.shape) == 2:
            # If 2D, assume it's (samples, features) and reshape
            motion_data = motion_data.reshape(-1, 6, 100)
        elif len(motion_data.shape) == 3:
            # If 3D, ensure correct shape (batch, channels, seq_len)
            if motion_data.shape[1] != 6:
                motion_data = motion_data.transpose(0, 2, 1)

        batch_size, channels, seq_len = motion_data.shape
        motion_normalized = np.zeros_like(motion_data)

        for i in range(channels):
            channel_data = motion_data[:, i, :].reshape(-1, seq_len)
            motion_normalized[:, i, :] = self.motion_scaler.fit_transform(channel_data)
        return torch.tensor(motion_normalized, dtype = torch.float32)
    
    def preprocess_hr_data(self, hr_data):
        # Handle different input shapes
        if len(hr_data.shape) == 1:
            # If 1D, reshape to (1, seq_len, 1)
            hr_data = hr_data.reshape(1, -1, 1)
        elif len(hr_data.shape) == 2:
            # If 2D, reshape to (batch, seq_len, 1)
            hr_data = hr_data.reshape(-1, self.hr_seq_len, 1)
        elif len(hr_data.shape) == 3:
            # If 3D, ensure correct shape (batch, seq_len, 1)
            if hr_data.shape[2] != 1:
                hr_data = hr_data.reshape(-1, self.hr_seq_len, 1)
        
        # Flatten for StandardScaler, then reshape back
        hr_flat = hr_data.reshape(-1, self.hr_seq_len)
        hr_normalized = self.hr_scaler.fit_transform(hr_flat)
        hr_reshaped = hr_normalized.reshape(-1, self.hr_seq_len, 1)

        return torch.tensor(hr_reshaped, dtype = torch.float32)
    
    def create_fusion_sequences(self, motion_data, hr_data, labels, seq_len = None):
        """Create sequences for fusion training - simplified approach"""
        if seq_len is None:
            seq_len = self.fusion_seq_len
        
        min_samples = min(len(motion_data), len(hr_data), len(labels))

        # For simplicity, let's just use the first seq_len samples from each
        # This avoids the complex overlapping sequence creation
        X_motion = motion_data[:min_samples]  # Shape: (min_samples, 6, 100)
        X_hr = hr_data[:min_samples]          # Shape: (min_samples, 100, 1)
        y = labels[:min_samples]              # Shape: (min_samples,)
        
        return (X_motion, X_hr, torch.tensor(y, dtype=torch.long))
    
    # NEW: Beach-specific methods
    def create_beach_dataset(self, num_samples=1000, seed=42):
        """Generate beach-specific dataset with environmental context"""
        np.random.seed(seed)
        
        # Generate base data (your existing method)
        motion_data, hr_data, labels = self.create_fusion_dataset(num_samples)
        
        # NEW: Add environmental data
        environmental_data = []
        for i in range(num_samples):
            # Simulate beach environmental conditions
            wave_height = np.random.uniform(0.1, 2.0)  # meters
            current_strength = np.random.uniform(0.1, 1.5)  # m/s
            wind_speed = np.random.uniform(0, 15)  # m/s
            water_temp = np.random.uniform(15, 30)  # Celsius
            
            env_sample = [wave_height, current_strength, wind_speed, water_temp]
            environmental_data.append(env_sample)
        
        environmental_data = np.array(environmental_data)
        
        return motion_data, hr_data, environmental_data, labels
    
    def preprocess_environmental_data(self, env_data):
        """Preprocess environmental data for beach context"""
        env_normalized = self.environmental_scaler.fit_transform(env_data)
        return torch.tensor(env_normalized, dtype=torch.float32)
    
    def classify_beach_activity(self, motion_data, hr_data):
        """Classify beach activities to reduce false alarms"""
        activities = []
        
        for i in range(len(motion_data)):
            motion_pattern = motion_data[i]
            hr_pattern = hr_data[i]
            
            # Simple activity classification based on patterns
            motion_variance = np.var(motion_pattern)
            hr_mean = np.mean(hr_pattern)
            
            if motion_variance > 2.0 and hr_mean > 100:
                activity = 'wave_riding'  # High motion, high HR
            elif motion_variance > 1.5 and hr_mean > 80:
                activity = 'swimming'     # Moderate motion, moderate HR
            elif motion_variance < 0.5 and hr_mean < 70:
                activity = 'floating'     # Low motion, low HR
            else:
                activity = 'unknown'
            
            activities.append(activity)
        
        return activities
            
    def save_fusion_data(self, motion_data, hr_data, labels,
                        motion_file =  "fusion_motion_data.csv",
                        hr_file = "fusion_hr_data.csv",
                        labels_file = "fusion_labels.csv"):
        #Motion data - reshape to 2D for CSV
        motion_df = pd.DataFrame(motion_data.reshape(motion_data.shape[0], -1))
        motion_df.to_csv(motion_file, index = False)
        print(f"Saved motion data to {motion_file}")

        #HR data - reshape to 2D for CSV
        hr_df = pd.DataFrame(hr_data.reshape(hr_data.shape[0], -1))
        hr_df.to_csv(hr_file, index = False)
        print(f"Saved heart rate data to {hr_file}")

        #Labels
        labels_df = pd.DataFrame(labels, columns = ['fusion_label'])
        labels_df.to_csv(labels_file, index = False)
        print(f"Saved labels to {labels_file}")

    def load_fusion_data(self, motion_file = "fusion_motion_data.csv",
                        hr_file = "fusion_hr_data.csv",
                        labels_file = "fusion_labels.csv"):
        """Load preprocessed data from CSV files"""

        #Motion data
        motion_df = pd.read_csv(motion_file)
        motion_data = motion_df.values.reshape(-1, 6, self.motion_seq_len)

        #HR data
        hr_df = pd.read_csv(hr_file)
        hr_data = hr_df.values.reshape(-1, self.hr_seq_len, 1)

        #Labels
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['fusion_label'].values

        return (torch.tensor(motion_data, dtype = torch.float32),
                torch.tensor(hr_data, dtype = torch.float32),
                torch.tensor(labels, dtype = torch.long))

def main():
    """Main preprocessing pipeline"""
    print("=== Fusion AI Data Preprocessing ===")

    #Initialize preprocessor
    preprocessor = FusionDataProcessor()


    #Generate synthetic data
    print("Generating synthetic data...")
    motion_data, hr_data, labels = preprocessor.create_fusion_dataset(num_samples = 2000)

    #Preprocess data
    print("Preprocessing data...")
    motion_processed = preprocessor.preprocess_motion_data(motion_data)

    print("Preprocessing heart rate data...")
    hr_processed = preprocessor.preprocess_hr_data(hr_data)

    print("Creating fusion sequences...")
    X_motion, X_hr, y = preprocessor.create_fusion_sequences(motion_processed, hr_processed, labels)

    #Split data
    X_motion_train, X_motion_test, X_hr_train, X_hr_test, y_train, y_test = train_test_split(
        X_motion, X_hr, y, test_size = 0.2, random_state = 42, stratify = y
    )

    #Save processed data
    print("Saving processed data...")
    preprocessor.save_fusion_data(
        X_motion_train.numpy(), X_hr_train.numpy(), y_train.numpy(),
        "fusion_motion_train.csv", "fusion_hr_train.csv", "fusion_labels_train.csv"
    )
    
    preprocessor.save_fusion_data(
        X_motion_test.numpy(), X_hr_test.numpy(), y_test.numpy(),
        "fusion_motion_test.csv", "fusion_hr_test.csv", "fusion_labels_test.csv"
    )

    print(f"\nPreprocssesing complete!")
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"Motion data shape: {X_motion_train.shape}")
    print(f"Heart rate data shape: {X_hr_train.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

if __name__ == "__main__":
    main()
                

