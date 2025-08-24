import pandas as pd
import numpy as np
import torch
from data_adapter import FlexibleDataAdapter
from fusion_train import FusionAITrainer
from fusion_preproces import FusionDataProcessor
import os

class CSVProcessor:
    """Process CSV datasets for fusion AI"""
    
    def __init__(self):
        self.adapter = FlexibleDataAdapter()
        self.trainer = FusionAITrainer()
        self.preprocessor = FusionDataProcessor()
    
    def load_and_analyze_csv(self, csv_file_path):
        """Load and analyze CSV data"""
        print(f"=== Loading CSV: {csv_file_path} ===")
        
        try:
            df = pd.read_csv(csv_file_path)
            print(f"✅ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Columns: {list(df.columns)}")
            
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None
    
    def extract_data_columns(self, df):
        """Extract different types of data from CSV"""
        print("\n=== Extracting Data Columns ===")
        
        data = {
            'motion': None,
            'heart_rate': None,
            'environmental': None,
            'labels': None
        }
        
        # Extract motion data (accelerometer/gyroscope)
        motion_cols = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['accel', 'gyro', 'motion', 'x', 'y', 'z'])]
        if motion_cols:
            print(f"✅ Motion columns: {motion_cols}")
            data['motion'] = df[motion_cols].values
        else:
            print("❌ No motion data found")
        
        # Extract heart rate data
        hr_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in ['hr', 'heart', 'bpm', 'pulse'])]
        if hr_cols:
            print(f"✅ Heart rate columns: {hr_cols}")
            data['heart_rate'] = df[hr_cols].values
        else:
            print("❌ No heart rate data found")
        
        # Extract environmental data
        env_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['wave', 'current', 'wind', 'temp', 'environment'])]
        if env_cols:
            print(f"✅ Environmental columns: {env_cols}")
            data['environmental'] = df[env_cols].values
        else:
            print("❌ No environmental data found")
        
        # Extract labels
        label_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['label', 'class', 'target', 'risk', 'emergency'])]
        if label_cols:
            print(f"✅ Label columns: {label_cols}")
            data['labels'] = df[label_cols].values
        else:
            print("❌ No labels found")
        
        return data
    
    def process_for_prediction(self, csv_file_path):
        """Process CSV for prediction (no training)"""
        print(f"\n=== Processing CSV for Prediction ===")
        
        # Load CSV
        df = self.load_and_analyze_csv(csv_file_path)
        if df is None:
            return
        
        # Extract data
        data = self.extract_data_columns(df)
        
        # Process each row for prediction
        predictions = []
        
        for i in range(min(10, len(df))):  # Process first 10 rows as example
            print(f"\n📊 Processing row {i+1}:")
            
            # Extract row data
            motion_data = data['motion'][i] if data['motion'] is not None else None
            hr_data = data['heart_rate'][i] if data['heart_rate'] is not None else None
            env_data = data['environmental'][i] if data['environmental'] is not None else None
            
            # Adapt data using flexible adapter
            try:
                if motion_data is not None:
                    motion_adapted = self.adapter.adapt_motion_data(motion_data, 'csv_motion')
                else:
                    print("⚠️  No motion data - using synthetic data")
                    motion_adapted = self.adapter.adapt_motion_data(
                        np.random.normal(0, 1, (100, 6)), 'synthetic'
                    )
                
                if hr_data is not None:
                    hr_adapted = self.adapter.adapt_heart_rate_data(hr_data, 'csv_hr')
                else:
                    print("⚠️  No heart rate data - using default")
                    hr_adapted = self.adapter.adapt_heart_rate_data(None, 'missing')
                
                if env_data is not None:
                    env_adapted = self.adapter.adapt_environmental_data(env_data, 'csv_env')
                else:
                    print("⚠️  No environmental data - using defaults")
                    env_adapted = self.adapter.adapt_environmental_data(
                        {'wave_height': 0.5, 'current_strength': 0.3, 'wind_speed': 5.0, 'water_temp': 25.0}, 
                        'default'
                    )
                
                # Get prediction
                prediction, probs = self.trainer.predict_beach_drowning(
                    motion_adapted, hr_adapted, env_adapted
                )
                
                risk_level = self.trainer.class_names[prediction.item()]
                env_risk = self.trainer.calculate_environmental_risk(env_adapted).item()
                
                predictions.append({
                    'row': i+1,
                    'risk_level': risk_level,
                    'environmental_risk': env_risk,
                    'probabilities': probs.cpu().numpy()
                })
                
                print(f"   Risk Level: {risk_level}")
                print(f"   Environmental Risk: {env_risk:.2f}x")
                
            except Exception as e:
                print(f"   ❌ Error processing row {i+1}: {e}")
        
        return predictions
    
    def process_for_training(self, csv_file_path):
        """Process CSV for training (requires labels)"""
        print(f"\n=== Processing CSV for Training ===")
        
        # Load CSV
        df = self.load_and_analyze_csv(csv_file_path)
        if df is None:
            return
        
        # Extract data
        data = self.extract_data_columns(df)
        
        # Check if we have labels
        if data['labels'] is None:
            print("❌ No labels found - cannot train without labels")
            print("💡 Use 'process_for_prediction()' instead")
            return
        
        print("✅ Labels found - proceeding with training preparation")
        
        # This would require more complex processing to create training datasets
        # For now, we'll show what would be needed
        print("\n📋 Training Data Requirements:")
        print("   - Motion data: (samples, 6, 100)")
        print("   - Heart rate data: (samples, 100, 1)")
        print("   - Labels: (samples,)")
        print("   - Environmental data: (samples, 4)")
        
        print("\n🔧 To implement training with CSV:")
        print("   1. Create sequences from time series data")
        print("   2. Align motion, HR, and environmental data")
        print("   3. Convert labels to numeric format")
        print("   4. Split into train/test sets")
        print("   5. Use fusion_train.py with custom data loader")

def main():
    """Main function to process CSV datasets"""
    print("=== CSV Dataset Processor for Fusion AI ===")
    
    # Check for CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in current directory")
        print("📝 Please place your CSV file in this directory and run again")
        return
    
    print(f"📁 Found CSV files: {csv_files}")
    
    # Let user choose file
    if len(csv_files) == 1:
        csv_file = csv_files[0]
    else:
        print("\n📋 Choose CSV file to process:")
        for i, file in enumerate(csv_files):
            print(f"   {i+1}. {file}")
        
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(csv_files):
                csv_file = csv_files[choice]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid input")
            return
    
    # Initialize processor
    processor = CSVProcessor()
    
    # Ask user what they want to do
    print(f"\n🎯 What do you want to do with {csv_file}?")
    print("   1. Prediction only (no labels needed)")
    print("   2. Training (requires labels)")
    
    try:
        choice = int(input("\nEnter choice (1 or 2): "))
        
        if choice == 1:
            print(f"\n🚀 Processing {csv_file} for prediction...")
            predictions = processor.process_for_prediction(csv_file)
            
            if predictions:
                print(f"\n✅ Successfully processed {len(predictions)} rows")
                print("📊 Summary of predictions:")
                for pred in predictions:
                    print(f"   Row {pred['row']}: {pred['risk_level']} (Env Risk: {pred['environmental_risk']:.2f}x)")
        
        elif choice == 2:
            print(f"\n🚀 Processing {csv_file} for training...")
            processor.process_for_training(csv_file)
        
        else:
            print("❌ Invalid choice")
    
    except ValueError:
        print("❌ Invalid input")

if __name__ == "__main__":
    main()
