import pandas as pd
import numpy as np
import os

def convert_dat_to_csv(dat_file_path):
    """Convert .dat file to CSV format"""
    print(f"=== Converting .dat file: {dat_file_path} ===")
    
    try:
        # Read .dat file
        with open(dat_file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"📊 Found {len(lines)} lines in .dat file")
        
        # Parse data
        data_rows = []
        for i, line in enumerate(lines):
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
            
            # Show progress for large files
            if i % 100 == 0 and i > 0:
                print(f"   Processed {i} lines...")
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        print(f"📋 Data shape: {df.shape}")
        print(f"📊 Data types: {df.dtypes.tolist()}")
        
        # Add meaningful column names
        if len(df.columns) >= 6:
            # Assume first 6 columns are motion data
            column_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            
            # Add additional feature names
            for i in range(6, len(df.columns)):
                column_names.append(f'feature_{i}')
            
            df.columns = column_names
        else:
            # Generic column names
            df.columns = [f'feature_{i}' for i in range(len(df.columns))]
        
        # Save as CSV
        output_path = dat_file_path.replace('.dat', '_converted.csv')
        df.to_csv(output_path, index=False)
        
        print(f"✅ Converted to CSV: {output_path}")
        print(f"📊 Final shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📋 First 3 rows:")
        print(df.head(3))
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error converting .dat file: {e}")
        return None

def create_beach_dataset():
    """Create a realistic beach drowning dataset"""
    print("\n=== Creating Beach Drowning Dataset ===")
    
    np.random.seed(42)
    n_samples = 500
    
    beach_data = []
    
    # Generate different scenarios
    scenarios = {
        'normal_swimming': {'motion_variance': 0.8, 'hr_mean': 75, 'count': 200},
        'wave_riding': {'motion_variance': 1.5, 'hr_mean': 110, 'count': 100},
        'floating': {'motion_variance': 0.2, 'hr_mean': 65, 'count': 100},
        'playing': {'motion_variance': 1.2, 'hr_mean': 95, 'count': 75},
        'drowning': {'motion_variance': 2.5, 'hr_mean': 150, 'count': 25}
    }
    
    for scenario, params in scenarios.items():
        print(f"   Generating {scenario} data...")
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
    output_path = "beach_drowning_dataset.csv"
    beach_df.to_csv(output_path, index=False)
    
    print(f"✅ Created beach dataset: {output_path}")
    print(f"📊 Dataset shape: {beach_df.shape}")
    print(f"🎯 Labels: {beach_df['activity_label'].value_counts().to_dict()}")
    
    return output_path

def main():
    """Main function"""
    print("=== Dataset Converter for Fusion AI ===")
    
    # Find .dat files
    dat_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
    
    print(f"\n📁 Found .dat files: {dat_files}")
    
    if dat_files:
        print("\n🎯 What would you like to do?")
        print("1. Convert .dat file to CSV")
        print("2. Create beach drowning dataset")
        print("3. Both")
        
        try:
            choice = int(input("\nEnter choice (1-3): "))
            
            if choice == 1:
                if len(dat_files) == 1:
                    convert_dat_to_csv(dat_files[0])
                else:
                    print("Multiple .dat files found. Converting first one...")
                    convert_dat_to_csv(dat_files[0])
            
            elif choice == 2:
                create_beach_dataset()
            
            elif choice == 3:
                if len(dat_files) == 1:
                    convert_dat_to_csv(dat_files[0])
                else:
                    print("Multiple .dat files found. Converting first one...")
                    convert_dat_to_csv(dat_files[0])
                create_beach_dataset()
            
            else:
                print("❌ Invalid choice")
        
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ No .dat files found")
        print("Creating beach dataset instead...")
        create_beach_dataset()
    
    print("\n🎉 Dataset preparation complete!")
    print("📁 Check the current directory for your CSV files")

if __name__ == "__main__":
    main()
