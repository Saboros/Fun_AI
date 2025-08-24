import requests
import pandas as pd
import numpy as np
import os
import zipfile
from urllib.parse import urlparse
import warnings
warnings.filterwarnings("ignore")

class DatasetDownloader:
    """Download and convert real datasets for fusion AI"""
    
    def __init__(self):
        self.datasets_dir = "real_datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def download_uci_har(self):
        """Download UCI HAR dataset - perfect for motion classification"""
        print("=== Downloading UCI HAR Dataset ===")
        
        # UCI HAR dataset URLs
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        
        try:
            # Download dataset
            print("📥 Downloading UCI HAR dataset...")
            response = requests.get(train_url, stream=True)
            zip_path = os.path.join(self.datasets_dir, "uci_har.zip")
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            print("📂 Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.datasets_dir)
            
            # Convert to fusion AI format
            print("🔄 Converting to fusion AI format...")
            self.convert_uci_har_to_fusion_format()
            
            print("✅ UCI HAR dataset ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading UCI HAR: {e}")
            return False
    
    def convert_uci_har_to_fusion_format(self):
        """Convert UCI HAR data to fusion AI format"""
        uci_dir = os.path.join(self.datasets_dir, "UCI HAR Dataset")
        
        # Load training data
        train_x = pd.read_csv(os.path.join(uci_dir, "train", "X_train.txt"), sep=' ', header=None)
        train_y = pd.read_csv(os.path.join(uci_dir, "train", "y_train.txt"), header=None)
        
        # Load test data
        test_x = pd.read_csv(os.path.join(uci_dir, "test", "X_test.txt"), sep=' ', header=None)
        test_y = pd.read_csv(os.path.join(uci_dir, "test", "y_test.txt"), header=None)
        
        # Load activity labels
        activities = pd.read_csv(os.path.join(uci_dir, "activity_labels.txt"), sep=' ', header=None)
        activity_dict = dict(zip(activities[0], activities[1]))
        
        # Combine data
        all_data = pd.concat([train_x, test_x], ignore_index=True)
        all_labels = pd.concat([train_y, test_y], ignore_index=True)
        
        # Create fusion AI format
        fusion_data = []
        
        # Each row in UCI HAR is already a sequence (561 features)
        # We'll reshape it to match our format
        for i in range(len(all_data)):
            # Extract motion features (first 6 are accel/gyro)
            motion_features = all_data.iloc[i, :6].values
            
            # Create synthetic heart rate (since UCI HAR doesn't have HR)
            heart_rate = np.random.normal(80, 15)  # Normal distribution around 80 BPM
            
            # Create synthetic environmental data
            env_data = {
                'wave_height': np.random.uniform(0.1, 2.0),
                'current_strength': np.random.uniform(0.1, 1.5),
                'wind_speed': np.random.uniform(0, 15),
                'water_temp': np.random.uniform(15, 30)
            }
            
            # Get activity label
            activity_id = all_labels.iloc[i, 0]
            activity_name = activity_dict[activity_id]
            
            # Map to our categories
            if activity_name in ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']:
                fusion_label = 'normal_swimming'
            elif activity_name in ['SITTING', 'STANDING']:
                fusion_label = 'floating'
            elif activity_name == 'LAYING':
                fusion_label = 'drowning'  # Immobile = potential drowning
            else:
                fusion_label = 'normal_swimming'
            
            # Create row
            row = {
                'accel_x': motion_features[0],
                'accel_y': motion_features[1], 
                'accel_z': motion_features[2],
                'gyro_x': motion_features[3],
                'gyro_y': motion_features[4],
                'gyro_z': motion_features[5],
                'heart_rate': heart_rate,
                'wave_height': env_data['wave_height'],
                'current_strength': env_data['current_strength'],
                'wind_speed': env_data['wind_speed'],
                'water_temp': env_data['water_temp'],
                'activity_label': fusion_label,
                'original_activity': activity_name
            }
            
            fusion_data.append(row)
        
        # Save as CSV
        fusion_df = pd.DataFrame(fusion_data)
        output_path = os.path.join(self.datasets_dir, "uci_har_fusion_format.csv")
        fusion_df.to_csv(output_path, index=False)
        
        print(f"✅ Converted to fusion format: {output_path}")
        print(f"📊 Dataset shape: {fusion_df.shape}")
        print(f"🎯 Labels: {fusion_df['activity_label'].value_counts().to_dict()}")
    
    def create_synthetic_beach_dataset(self):
        """Create a realistic beach drowning dataset"""
        print("=== Creating Synthetic Beach Dataset ===")
        
        np.random.seed(42)
        n_samples = 1000
        
        beach_data = []
        
        # Generate different scenarios
        scenarios = {
            'normal_swimming': {'motion_variance': 0.8, 'hr_mean': 75, 'count': 400},
            'wave_riding': {'motion_variance': 1.5, 'hr_mean': 110, 'count': 200},
            'floating': {'motion_variance': 0.2, 'hr_mean': 65, 'count': 200},
            'playing': {'motion_variance': 1.2, 'hr_mean': 95, 'count': 150},
            'drowning': {'motion_variance': 2.5, 'hr_mean': 150, 'count': 50}
        }
        
        for scenario, params in scenarios.items():
            for _ in range(params['count']):
                # Generate motion data
                motion_variance = params['motion_variance']
                accel_x = np.random.normal(0, motion_variance)
                accel_y = np.random.normal(0, motion_variance)
                accel_z = np.random.normal(0, motion_variance)
                gyro_x = np.random.normal(0, motion_variance * 0.1)
                gyro_y = np.random.normal(0, motion_variance * 0.1)
                gyro_z = np.random.normal(0, motion_variance * 0.1)
                
                # Generate heart rate
                hr_mean = params['hr_mean']
                heart_rate = np.random.normal(hr_mean, 10)
                
                # Generate environmental data
                if scenario == 'drowning':
                    # Dangerous conditions for drowning
                    wave_height = np.random.uniform(1.5, 3.0)
                    current_strength = np.random.uniform(1.0, 2.0)
                    wind_speed = np.random.uniform(15, 25)
                    water_temp = np.random.uniform(15, 20)
                else:
                    # Normal conditions
                    wave_height = np.random.uniform(0.1, 1.5)
                    current_strength = np.random.uniform(0.1, 1.0)
                    wind_speed = np.random.uniform(0, 15)
                    water_temp = np.random.uniform(20, 30)
                
                row = {
                    'accel_x': accel_x,
                    'accel_y': accel_y,
                    'accel_z': accel_z,
                    'gyro_x': gyro_x,
                    'gyro_y': gyro_y,
                    'gyro_z': gyro_z,
                    'heart_rate': heart_rate,
                    'wave_height': wave_height,
                    'current_strength': current_strength,
                    'wind_speed': wind_speed,
                    'water_temp': water_temp,
                    'activity_label': scenario
                }
                
                beach_data.append(row)
        
        # Save dataset
        beach_df = pd.DataFrame(beach_data)
        output_path = os.path.join(self.datasets_dir, "synthetic_beach_dataset.csv")
        beach_df.to_csv(output_path, index=False)
        
        print(f"✅ Created beach dataset: {output_path}")
        print(f"📊 Dataset shape: {beach_df.shape}")
        print(f"🎯 Labels: {beach_df['activity_label'].value_counts().to_dict()}")
    
    def convert_dat_file(self, dat_file_path):
        """Convert .dat file to CSV format"""
        print(f"=== Converting .dat file: {dat_file_path} ===")
        
        try:
            # Read .dat file
            with open(dat_file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse data
            data_rows = []
            for line in lines:
                # Split by whitespace and convert to numbers
                values = line.strip().split()
                numeric_values = []
                
                for val in values:
                    try:
                        if val == 'NaN':
                            numeric_values.append(np.nan)
                        else:
                            numeric_values.append(float(val))
                    except ValueError:
                        continue
                
                if len(numeric_values) > 0:
                    data_rows.append(numeric_values)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Add column names
            if len(df.columns) >= 6:
                # Assume first 6 columns are motion data
                df.columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'] + \
                            [f'feature_{i}' for i in range(len(df.columns) - 6)]
            
            # Save as CSV
            output_path = dat_file_path.replace('.dat', '_converted.csv')
            df.to_csv(output_path, index=False)
            
            print(f"✅ Converted to CSV: {output_path}")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error converting .dat file: {e}")
            return None

def main():
    """Main function to download and convert datasets"""
    print("=== Real Dataset Downloader for Fusion AI ===")
    
    downloader = DatasetDownloader()
    
    print("\n🎯 Available options:")
    print("1. Download UCI HAR dataset (motion classification)")
    print("2. Create synthetic beach dataset (drowning detection)")
    print("3. Convert your .dat file to CSV")
    print("4. All of the above")
    
    try:
        choice = int(input("\nEnter choice (1-4): "))
        
        if choice == 1:
            downloader.download_uci_har()
        elif choice == 2:
            downloader.create_synthetic_beach_dataset()
        elif choice == 3:
            # Find .dat files
            dat_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.dat'):
                        dat_files.append(os.path.join(root, file))
            
            if dat_files:
                print(f"\n📁 Found .dat files: {dat_files}")
                if len(dat_files) == 1:
                    downloader.convert_dat_file(dat_files[0])
                else:
                    print("Multiple .dat files found. Please specify which one to convert.")
            else:
                print("❌ No .dat files found in current directory")
        elif choice == 4:
            downloader.download_uci_har()
            downloader.create_synthetic_beach_dataset()
            # Convert .dat files if found
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.dat'):
                        downloader.convert_dat_file(os.path.join(root, file))
        else:
            print("❌ Invalid choice")
    
    except ValueError:
        print("❌ Invalid input")
    
    print("\n🎉 Dataset preparation complete!")
    print("📁 Check the 'real_datasets' folder for your converted data")

if __name__ == "__main__":
    main()
