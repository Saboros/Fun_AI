import numpy as np
import pandas as pd
import torch
from typing import Union, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class FlexibleDataAdapter:
    """Adapts real-world sensor data to the fusion AI format"""
    
    def __init__(self, target_seq_len=100, target_motion_channels=6, target_hr_features=1):
        self.target_seq_len = target_seq_len
        self.target_motion_channels = target_motion_channels
        self.target_hr_features = target_hr_features
        
    def adapt_motion_data(self, data: Union[np.ndarray, pd.DataFrame, List], 
                         data_type: str = 'auto') -> torch.Tensor:
        """
        Adapt motion data from various formats to (batch, channels, timesteps)
        
        Args:
            data: Motion data in various formats
            data_type: 'auto', 'csv', 'json', 'numpy', 'list'
        """
        print(f"Adapting motion data of type: {data_type}")
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        # Handle different input shapes
        if len(data.shape) == 1:
            # Single channel, single timestep
            data = data.reshape(1, 1, 1)
        elif len(data.shape) == 2:
            # (timesteps, channels) or (channels, timesteps)
            if data.shape[0] < data.shape[1]:
                # (channels, timesteps) -> (1, channels, timesteps)
                data = data.reshape(1, data.shape[0], data.shape[1])
            else:
                # (timesteps, channels) -> (1, channels, timesteps)
                data = data.T.reshape(1, data.shape[1], data.shape[0])
        elif len(data.shape) == 3:
            # Already in (batch, channels, timesteps) format
            pass
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Handle different channel configurations
        current_channels = data.shape[1]
        
        if current_channels == 3:
            # Only accelerometer data - duplicate for gyroscope
            print("3-axis accelerometer detected - duplicating for gyroscope")
            data = np.concatenate([data, data], axis=1)
        elif current_channels == 9:
            # 9-axis IMU - select 6 channels (3 accel + 3 gyro)
            print("9-axis IMU detected - selecting 6 channels")
            data = data[:, :6, :]
        elif current_channels != 6:
            # Other configurations - pad or truncate
            print(f"Non-standard channels ({current_channels}) - adapting to 6 channels")
            if current_channels < 6:
                # Pad with zeros
                padding = np.zeros((data.shape[0], 6 - current_channels, data.shape[2]))
                data = np.concatenate([data, padding], axis=1)
            else:
                # Truncate to 6 channels
                data = data[:, :6, :]
        
        # Handle different sequence lengths
        current_timesteps = data.shape[2]
        
        if current_timesteps != self.target_seq_len:
            print(f"Resampling from {current_timesteps} to {self.target_seq_len} timesteps")
            data = self._resample_sequence(data, self.target_seq_len)
        
        return torch.tensor(data, dtype=torch.float32)
    
    def adapt_heart_rate_data(self, data: Union[np.ndarray, pd.DataFrame, List, float, None], 
                             data_type: str = 'auto') -> torch.Tensor:
        """
        Adapt heart rate data from various formats to (batch, timesteps, features)
        
        Args:
            data: Heart rate data in various formats
            data_type: 'auto', 'csv', 'json', 'numpy', 'list', 'single_value', 'missing'
        """
        print(f"Adapting heart rate data of type: {data_type}")
        
        # Handle missing data
        if data is None:
            print("Missing heart rate data - using default values")
            data = np.full(self.target_seq_len, 75.0)  # Default normal heart rate
        
        # Handle single value
        if isinstance(data, (int, float)):
            print("Single heart rate value detected - creating sequence")
            data = np.full(self.target_seq_len, data)
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        elif isinstance(data, list):
            data = np.array(data)
        
        # Handle different input shapes
        if len(data.shape) == 1:
            # Single sequence
            if len(data) != self.target_seq_len:
                print(f"Resampling heart rate from {len(data)} to {self.target_seq_len} timesteps")
                data = self._resample_1d_sequence(data, self.target_seq_len)
            data = data.reshape(1, self.target_seq_len, 1)
        elif len(data.shape) == 2:
            # Multiple sequences or (timesteps, features)
            if data.shape[1] == 1:
                # (timesteps, 1) -> (1, timesteps, 1)
                data = data.reshape(1, data.shape[0], 1)
            else:
                # (batch, timesteps) -> (batch, timesteps, 1)
                data = data.reshape(data.shape[0], data.shape[1], 1)
        elif len(data.shape) == 3:
            # Already in (batch, timesteps, features) format
            pass
        else:
            raise ValueError(f"Unsupported heart rate data shape: {data.shape}")
        
        # Handle different sequence lengths
        current_timesteps = data.shape[1]
        if current_timesteps != self.target_seq_len:
            print(f"Resampling heart rate from {current_timesteps} to {self.target_seq_len} timesteps")
            data = self._resample_3d_sequence(data, self.target_seq_len)
        
        return torch.tensor(data, dtype=torch.float32)
    
    def adapt_environmental_data(self, data: Union[np.ndarray, pd.DataFrame, List, Dict], 
                                data_type: str = 'auto') -> torch.Tensor:
        """
        Adapt environmental data to (batch, features) format
        
        Args:
            data: Environmental data in various formats
            data_type: 'auto', 'csv', 'json', 'numpy', 'list', 'dict'
        """
        print(f"Adapting environmental data of type: {data_type}")
        
        # Handle dictionary format
        if isinstance(data, dict):
            print("Dictionary format detected - extracting values")
            # Expected keys: wave_height, current_strength, wind_speed, water_temp
            expected_keys = ['wave_height', 'current_strength', 'wind_speed', 'water_temp']
            values = []
            for key in expected_keys:
                if key in data:
                    values.append(data[key])
                else:
                    print(f"Missing {key} - using default value 0")
                    values.append(0.0)
            data = np.array(values)
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        # Handle different shapes
        if len(data.shape) == 1:
            # Single sample
            data = data.reshape(1, -1)
        elif len(data.shape) == 2:
            # Multiple samples
            pass
        else:
            raise ValueError(f"Unsupported environmental data shape: {data.shape}")
        
        # Handle different feature counts
        current_features = data.shape[1]
        if current_features != 4:
            print(f"Non-standard features ({current_features}) - adapting to 4 features")
            if current_features < 4:
                # Pad with zeros
                padding = np.zeros((data.shape[0], 4 - current_features))
                data = np.concatenate([data, padding], axis=1)
            else:
                # Truncate to 4 features
                data = data[:, :4]
        
        return torch.tensor(data, dtype=torch.float32)
    
    def _resample_sequence(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample 3D sequence data to target length"""
        batch_size, channels, current_length = data.shape
        
        if current_length == target_length:
            return data
        
        # Simple linear interpolation
        resampled = np.zeros((batch_size, channels, target_length))
        
        for b in range(batch_size):
            for c in range(channels):
                # Create interpolation indices
                old_indices = np.linspace(0, current_length - 1, current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                
                # Interpolate
                resampled[b, c, :] = np.interp(new_indices, old_indices, data[b, c, :])
        
        return resampled
    
    def _resample_1d_sequence(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample 1D sequence data to target length"""
        current_length = len(data)
        
        if current_length == target_length:
            return data
        
        # Simple linear interpolation
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, target_length)
        
        return np.interp(new_indices, old_indices, data)
    
    def _resample_3d_sequence(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample 3D sequence data to target length"""
        batch_size, current_length, features = data.shape
        
        if current_length == target_length:
            return data
        
        # Simple linear interpolation
        resampled = np.zeros((batch_size, target_length, features))
        
        for b in range(batch_size):
            for f in range(features):
                # Create interpolation indices
                old_indices = np.linspace(0, current_length - 1, current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                
                # Interpolate
                resampled[b, :, f] = np.interp(new_indices, old_indices, data[b, :, f])
        
        return resampled
    
    def create_sample_data(self) -> Dict[str, torch.Tensor]:
        """Create sample data in various formats for testing"""
        print("Creating sample data in various formats...")
        
        # Sample 1: 3-axis accelerometer data (50 timesteps)
        accel_3d = np.random.normal(0, 1, (50, 3))
        
        # Sample 2: 9-axis IMU data (200 timesteps)
        imu_9d = np.random.normal(0, 1, (200, 9))
        
        # Sample 3: Single heart rate value
        hr_single = 85.0
        
        # Sample 4: Heart rate sequence (75 timesteps)
        hr_sequence = np.random.normal(80, 10, 75)
        
        # Sample 5: Environmental data as dictionary
        env_dict = {
            'wave_height': 1.2,
            'current_strength': 0.8,
            'wind_speed': 12.0,
            'water_temp': 22.0
        }
        
        # Sample 6: Environmental data as array
        env_array = np.array([0.5, 0.3, 5.0, 25.0])
        
        return {
            'accel_3d': accel_3d,
            'imu_9d': imu_9d,
            'hr_single': hr_single,
            'hr_sequence': hr_sequence,
            'env_dict': env_dict,
            'env_array': env_array
        }

def test_data_adapter():
    """Test the flexible data adapter with various data formats"""
    print("=== Testing Flexible Data Adapter ===")
    
    adapter = FlexibleDataAdapter()
    sample_data = adapter.create_sample_data()
    
    # Test motion data adaptation
    print("\n--- Motion Data Adaptation ---")
    
    # Test 3-axis accelerometer
    motion_3d = adapter.adapt_motion_data(sample_data['accel_3d'], '3d_accel')
    print(f"3-axis accel shape: {motion_3d.shape}")
    
    # Test 9-axis IMU
    motion_9d = adapter.adapt_motion_data(sample_data['imu_9d'], '9d_imu')
    print(f"9-axis IMU shape: {motion_9d.shape}")
    
    # Test heart rate adaptation
    print("\n--- Heart Rate Adaptation ---")
    
    # Test single value
    hr_single = adapter.adapt_heart_rate_data(sample_data['hr_single'], 'single')
    print(f"Single HR shape: {hr_single.shape}")
    
    # Test sequence
    hr_seq = adapter.adapt_heart_rate_data(sample_data['hr_sequence'], 'sequence')
    print(f"HR sequence shape: {hr_seq.shape}")
    
    # Test environmental adaptation
    print("\n--- Environmental Data Adaptation ---")
    
    # Test dictionary
    env_dict = adapter.adapt_environmental_data(sample_data['env_dict'], 'dict')
    print(f"Environmental dict shape: {env_dict.shape}")
    
    # Test array
    env_array = adapter.adapt_environmental_data(sample_data['env_array'], 'array')
    print(f"Environmental array shape: {env_array.shape}")
    
    print("\n✅ All data adaptation tests passed!")

if __name__ == "__main__":
    test_data_adapter()
