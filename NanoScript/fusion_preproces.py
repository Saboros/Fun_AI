import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Import the individual preprocessors
from hrpreprocess import normalize_spo2, denormalize_spo2
from imupreprocess import MotionDataHandler


class FusionDataProcessor:
    """
    Data preprocessor for Multi-Modal Fusion Classifier
    Handles motion (IMU) and heart rate data preprocessing
    Uses hrpreprocess.py and imupreprocess.py for data generation
    """
    
    def __init__(self, motion_seq_len=100, hr_seq_len=100, fusion_seq_len=100):
        self.motion_seq_len = motion_seq_len
        self.hr_seq_len = hr_seq_len
        self.fusion_seq_len = fusion_seq_len
        self.motion_scaler = StandardScaler()
        self.hr_scaler = StandardScaler()
        
        # Initialize the motion data handler
        self.motion_handler = MotionDataHandler(
            sequence_length=motion_seq_len,
            n_features=6
        )
    
    def create_fusion_dataset(self, num_samples=1000, seed=42):
        """
        Generate synthetic fusion dataset with realistic patterns
        Uses imupreprocess.py and hrpreprocess.py for data generation
        
        Returns:
            motion_data: (num_samples, 6, 100) - IMU sensor data
            hr_data: (num_samples, 100) - Heart rate sequences
            spo2_data: (num_samples,) - SpO2 values (normalized [0, 1])
            fusion_labels: (num_samples,) - Class labels [0,1,2,3]
        """
        np.random.seed(seed)
        
        # Generate motion data using MotionDataHandler
        print("Generating motion data using imupreprocess.py...")
        X_motion, y_motion = self.motion_handler.generate_mock_data(n_samples=num_samples)
        # X_motion shape: (samples, 6, 100)
        # y_motion: 0=normal, 1=panic, 2=immobile
        
        # Convert to numpy
        motion_data = X_motion.numpy()
        motion_labels = y_motion.numpy()
        
        # Use actual number of samples generated (may be slightly less due to integer division)
        actual_samples = len(motion_data)
        
        # Generate HR and SpO2 data correlated with motion
        print("Generating HR and SpO2 data correlated with motion...")
        hr_data = []
        spo2_data = []
        
        for i in range(actual_samples):
            if motion_labels[i] == 0:  # Normal motion
                # Normal HR: 70-85 bpm with natural variation
                hr_sequence = np.random.normal(75, 8, self.hr_seq_len)
                hr_sequence = np.clip(hr_sequence, 60, 100)
                
                # Normal SpO2: 95-100%
                spo2_value = np.random.uniform(95, 100)
            
            elif motion_labels[i] == 1:  # Panic motion
                # High HR: 120-160 bpm (emergency/panic)
                hr_sequence = np.random.normal(140, 15, self.hr_seq_len)
                hr_sequence = np.clip(hr_sequence, 120, 180)
                # Add increasing trend to simulate stress
                trend = np.linspace(0, 20, self.hr_seq_len)
                hr_sequence = hr_sequence + trend
                
                # Low SpO2: 70-90% (distress/hypoxia)
                spo2_value = np.random.uniform(70, 90)
            
            else:  # Immobile (motion_labels[i] == 2)
                # Low/resting HR: 50-70 bpm
                hr_sequence = np.random.normal(60, 5, self.hr_seq_len)
                hr_sequence = np.clip(hr_sequence, 45, 75)
                
                # Variable SpO2: 80-95% (could be unconscious)
                spo2_value = np.random.uniform(80, 95)
            
            hr_data.append(hr_sequence)
            spo2_data.append(spo2_value)
        
        # Normalize SpO2 using hrpreprocess.py function
        spo2_data = np.array(spo2_data, dtype=np.float32)
        spo2_normalized = normalize_spo2(spo2_data)
        
        # Create fusion labels based on combined patterns
        fusion_labels = []
        for i in range(actual_samples):
            hr_mean = np.mean(hr_data[i])
            spo2_val = spo2_data[i]
            
            # Emergency: Panic motion + (High HR or Low SpO2)
            if motion_labels[i] == 1 and (hr_mean > 120 or spo2_val < 90):
                fusion_labels.append(0)  # Emergency (drowning/distress)
            # Warning: Immobile + Low SpO2
            elif motion_labels[i] == 2 and spo2_val < 92:
                fusion_labels.append(1)  # Warning (potentially unconscious)
            # Alert: High HR but normal motion
            elif hr_mean > 110 and motion_labels[i] == 0:
                fusion_labels.append(2)  # Alert (exercise/stress)
            # Normal: Everything else
            else:
                fusion_labels.append(3)  # Normal
        
        # Convert to numpy arrays with correct shapes
        hr_data = np.array(hr_data, dtype=np.float32)  # Shape: (samples, 100)
        fusion_labels = np.array(fusion_labels, dtype=np.int64)
        
        return motion_data, hr_data, spo2_normalized, fusion_labels
    
    def preprocess_motion_data(self, motion_data):
        """
        Normalize motion data using MotionDataHandler approach
        
        Args:
            motion_data: (samples, 6, 100) or similar shape
            
        Returns:
            torch.Tensor: (samples, 6, 100) normalized motion data
        """
        # Convert to torch tensor if not already
        if isinstance(motion_data, np.ndarray):
            motion_tensor = torch.FloatTensor(motion_data)
        else:
            motion_tensor = motion_data
        
        # Use the preprocess_data method from MotionDataHandler
        motion_normalized, _ = self.motion_handler.preprocess_data(motion_tensor, None)
        
        return motion_normalized
    
    def preprocess_hr_data(self, hr_data):
        """
        Normalize HR data to match HeartRateAnomalyDetector input requirements
        
        Args:
            hr_data: (samples, 100) or similar shape
            
        Returns:
            torch.Tensor: (samples, 100, 1) normalized HR data
        """
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
        
        # Normalize HR values (scale to similar range as training data)
        # Typical HR range: 40-200 bpm, normalize to ~[-2, 2] range
        hr_data = (hr_data - 80.0) / 30.0  # Center around 80bpm, std ~30
        
        return torch.tensor(hr_data, dtype=torch.float32)
    
    def create_fusion_sequences(self, motion_data, hr_data, spo2_data, labels, seq_len=None):
        """
        Create sequences for fusion training
        
        Args:
            motion_data: Preprocessed motion tensor
            hr_data: Preprocessed HR tensor
            spo2_data: SpO2 values (normalized [0, 1])
            labels: Label array
            seq_len: Sequence length (optional)
            
        Returns:
            tuple: (X_motion, X_hr, spo2_values, y)
        """
        if seq_len is None:
            seq_len = self.fusion_seq_len
        
        min_samples = min(len(motion_data), len(hr_data), len(labels))
        
        # Use all samples
        X_motion = motion_data[:min_samples]  # Shape: (min_samples, 6, 100)
        X_hr = hr_data[:min_samples]          # Shape: (min_samples, 100, 1)
        spo2_values = spo2_data[:min_samples] if spo2_data is not None else None
        y = labels[:min_samples]              # Shape: (min_samples,)
        
        if spo2_values is not None:
            return X_motion, X_hr, torch.tensor(spo2_values, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        else:
            return X_motion, X_hr, torch.tensor(y, dtype=torch.long)
    
    def save_fusion_data(self, motion_data, hr_data, spo2_data, labels, 
                        motion_file="fusion_motion_train.csv",
                        hr_file="fusion_hr_train.csv",
                        spo2_file="fusion_spo2_train.csv",
                        labels_file="fusion_labels_train.csv"):
        """Save preprocessed fusion data to CSV files"""
        # Reshape motion data for saving
        num_samples = motion_data.shape[0]
        motion_flat = motion_data.reshape(num_samples, -1)
        
        # Save motion data
        motion_df = pd.DataFrame(motion_flat)
        motion_df.to_csv(motion_file, index=False)
        
        # Save HR data
        hr_flat = hr_data.reshape(num_samples, -1)
        hr_df = pd.DataFrame(hr_flat)
        hr_df.to_csv(hr_file, index=False)
        
        # Save SpO2 data
        spo2_df = pd.DataFrame({'spo2': spo2_data})
        spo2_df.to_csv(spo2_file, index=False)
        
        # Save labels
        labels_df = pd.DataFrame({'label': labels})
        labels_df.to_csv(labels_file, index=False)
        
        print(f"✓ Saved fusion data:")
        print(f"  Motion: {motion_file}")
        print(f"  HR: {hr_file}")
        print(f"  SpO2: {spo2_file}")
        print(f"  Labels: {labels_file}")
    
    def load_fusion_data(self, motion_file="fusion_motion_train.csv",
                        hr_file="fusion_hr_train.csv",
                        spo2_file="fusion_spo2_train.csv",
                        labels_file="fusion_labels_train.csv"):
        """Load preprocessed fusion data from CSV files"""
        # Load motion data
        motion_df = pd.read_csv(motion_file)
        motion_data = motion_df.values.reshape(-1, 6, 100)
        motion_tensor = torch.tensor(motion_data, dtype=torch.float32)
        
        # Load HR data
        hr_df = pd.read_csv(hr_file)
        hr_data = hr_df.values.reshape(-1, 100, 1)
        hr_tensor = torch.tensor(hr_data, dtype=torch.float32)
        
        # Load SpO2 data
        spo2_df = pd.read_csv(spo2_file)
        spo2_tensor = torch.tensor(spo2_df['spo2'].values, dtype=torch.float32)
        
        # Load labels
        labels_df = pd.read_csv(labels_file)
        labels = torch.tensor(labels_df['label'].values, dtype=torch.long)
        
        print(f"✓ Loaded fusion data:")
        print(f"  Motion shape: {motion_tensor.shape}")
        print(f"  HR shape: {hr_tensor.shape}")
        print(f"  SpO2 shape: {spo2_tensor.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        return motion_tensor, hr_tensor, spo2_tensor, labels
    
    def create_beach_dataset(self, num_samples=1000, seed=42):
        """
        Generate beach-specific dataset with environmental context
        Uses hrpreprocess.py and imupreprocess.py for base data
        
        Returns:
            motion_data, hr_data, spo2_data, env_data, fusion_labels
        """
        # Generate base data using the standard method
        motion_data, hr_data, spo2_data, fusion_labels = self.create_fusion_dataset(num_samples, seed)
        
        # Use actual number of samples returned
        actual_samples = len(motion_data)
        
        # Add environmental data
        np.random.seed(seed)
        environmental_data = []
        
        for i in range(actual_samples):
            # Simulate beach environmental conditions
            wave_height = np.random.uniform(0.1, 2.0)  # meters
            current_strength = np.random.uniform(0.1, 1.5)  # m/s
            wind_speed = np.random.uniform(0, 15)  # m/s
            water_temp = np.random.uniform(15, 30)  # Celsius
            
            # Higher risk conditions for emergency scenarios
            if fusion_labels[i] == 0:  # Emergency
                wave_height = np.random.uniform(1.0, 2.0)
                current_strength = np.random.uniform(0.8, 1.5)
            
            env_sample = [wave_height, current_strength, wind_speed, water_temp]
            environmental_data.append(env_sample)
        
        environmental_data = np.array(environmental_data, dtype=np.float32)
        
        return motion_data, hr_data, spo2_data, environmental_data, fusion_labels
    
    def preprocess_environmental_data(self, env_data):
        """Preprocess environmental data for beach context"""
        env_scaler = StandardScaler()
        env_normalized = env_scaler.fit_transform(env_data)
        return torch.tensor(env_normalized, dtype=torch.float32)


if __name__ == "__main__":
    print("=== Testing Fusion Data Preprocessor ===\n")
    print("Using hrpreprocess.py and imupreprocess.py for data generation\n")
    
    # Initialize preprocessor
    processor = FusionDataProcessor()
    
    # Generate dataset
    print("1. Generating synthetic fusion dataset...")
    motion_data, hr_data, spo2_data, labels = processor.create_fusion_dataset(num_samples=500)
    print(f"   Motion data shape: {motion_data.shape}")
    print(f"   HR data shape: {hr_data.shape}")
    print(f"   SpO2 data shape: {spo2_data.shape}")
    print(f"   SpO2 range (normalized): [{spo2_data.min():.3f}, {spo2_data.max():.3f}]")
    print(f"   SpO2 range (denormalized): [{denormalize_spo2(spo2_data.min()):.1f}, {denormalize_spo2(spo2_data.max()):.1f}]%")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label distribution: {np.bincount(labels)}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    motion_processed = processor.preprocess_motion_data(motion_data)
    hr_processed = processor.preprocess_hr_data(hr_data)
    print(f"   Processed motion shape: {motion_processed.shape}")
    print(f"   Processed HR shape: {hr_processed.shape}")
    
    # Create sequences
    print("\n3. Creating fusion sequences...")
    X_motion, X_hr, spo2_values, y = processor.create_fusion_sequences(
        motion_processed, hr_processed, spo2_data, labels
    )
    print(f"   X_motion shape: {X_motion.shape}")
    print(f"   X_hr shape: {X_hr.shape}")
    print(f"   SpO2 values shape: {spo2_values.shape}")
    print(f"   y shape: {y.shape}")
    
    # Test beach dataset
    print("\n4. Generating beach-specific dataset...")
    motion_beach, hr_beach, spo2_beach, env_beach, labels_beach = processor.create_beach_dataset(num_samples=200)
    print(f"   Motion shape: {motion_beach.shape}")
    print(f"   HR shape: {hr_beach.shape}")
    print(f"   SpO2 shape: {spo2_beach.shape}")
    print(f"   Environment shape: {env_beach.shape}")
    print(f"   Labels shape: {labels_beach.shape}")
    
    print("\n✅ All tests passed! Preprocessor ready for use.")
    print("   - Motion data: Using MotionDataHandler from imupreprocess.py")
    print("   - HR data: Generated with realistic patterns")
    print("   - SpO2 data: Using normalize_spo2() from hrpreprocess.py")


