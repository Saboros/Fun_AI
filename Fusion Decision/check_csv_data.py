import pandas as pd
import numpy as np
import os

def check_csv_data(csv_file_path):
    """Check CSV data format and provide guidance"""
    print(f"=== Checking CSV Data: {csv_file_path} ===")
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file_path)
        
        print(f"\n📊 CSV Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.tolist()}")
        
        print(f"\n📋 First 5 rows:")
        print(df.head())
        
        print(f"\n📈 Data Statistics:")
        print(df.describe())
        
        # Analyze data structure
        print(f"\n🔍 Data Analysis:")
        
        # Check for motion data (accelerometer/gyroscope)
        motion_columns = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['accel', 'gyro', 'motion', 'x', 'y', 'z'])]
        if motion_columns:
            print(f"✅ Motion data detected: {motion_columns}")
        else:
            print("❌ No motion data columns found")
        
        # Check for heart rate data
        hr_columns = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['hr', 'heart', 'bpm', 'pulse'])]
        if hr_columns:
            print(f"✅ Heart rate data detected: {hr_columns}")
        else:
            print("❌ No heart rate data columns found")
        
        # Check for environmental data
        env_columns = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['wave', 'current', 'wind', 'temp', 'environment'])]
        if env_columns:
            print(f"✅ Environmental data detected: {env_columns}")
        else:
            print("❌ No environmental data columns found")
        
        # Check for labels
        label_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['label', 'class', 'target', 'risk', 'emergency'])]
        if label_columns:
            print(f"✅ Label data detected: {label_columns}")
        else:
            print("❌ No label columns found")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def suggest_processing_steps(df):
    """Suggest how to process the CSV data"""
    print(f"\n🎯 Suggested Processing Steps:")
    
    if df is None:
        print("❌ Cannot suggest steps - CSV not loaded")
        return
    
    # Check what data we have
    has_motion = any('accel' in col.lower() or 'gyro' in col.lower() for col in df.columns)
    has_hr = any('hr' in col.lower() or 'heart' in col.lower() for col in df.columns)
    has_env = any('wave' in col.lower() or 'current' in col.lower() for col in df.columns)
    has_labels = any('label' in col.lower() or 'class' in col.lower() for col in df.columns)
    
    print(f"\n📱 Data Available:")
    print(f"   Motion: {'✅' if has_motion else '❌'}")
    print(f"   Heart Rate: {'✅' if has_hr else '❌'}")
    print(f"   Environmental: {'✅' if has_env else '❌'}")
    print(f"   Labels: {'✅' if has_labels else '❌'}")
    
    print(f"\n🚀 Next Steps:")
    
    if has_motion and has_hr and has_labels:
        print("1. ✅ Use 'process_csv_dataset.py' for full training")
    elif has_motion and has_hr:
        print("1. ✅ Use 'real_world_example.py' for prediction only")
    elif has_motion or has_hr:
        print("1. ⚠️  Use 'real_world_example.py' with missing data handling")
    else:
        print("1. ❌ Data format not suitable for fusion AI")
    
    print("2. 📊 Run 'check_csv_data.py' to verify data format")
    print("3. 🔧 Modify data if needed")
    print("4. 🎯 Run appropriate processing script")

def main():
    """Main function to check CSV data"""
    print("=== CSV Data Checker for Fusion AI ===")
    
    # Check if CSV file exists in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if csv_files:
        print(f"\n📁 Found CSV files: {csv_files}")
        
        if len(csv_files) == 1:
            csv_file = csv_files[0]
            print(f"\n🔍 Checking: {csv_file}")
            df = check_csv_data(csv_file)
            suggest_processing_steps(df)
        else:
            print(f"\n📋 Multiple CSV files found. Please specify which one to check:")
            for i, file in enumerate(csv_files):
                print(f"   {i+1}. {file}")
            
            try:
                choice = int(input("\nEnter number (or 0 to exit): ")) - 1
                if 0 <= choice < len(csv_files):
                    csv_file = csv_files[choice]
                    print(f"\n🔍 Checking: {csv_file}")
                    df = check_csv_data(csv_file)
                    suggest_processing_steps(df)
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Invalid input")
    else:
        print("❌ No CSV files found in current directory")
        print("\n📝 To use this script:")
        print("1. Place your CSV file in this directory")
        print("2. Run: python check_csv_data.py")
        print("3. Follow the suggested steps")

if __name__ == "__main__":
    main()
