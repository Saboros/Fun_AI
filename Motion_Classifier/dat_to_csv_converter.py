import pandas as pd
import numpy as np

def inspect_dat_file(dat_file_path: str):
   """Inspect the structure of the .dat file"""
   try:
       # Read first few lines of the file
       with open(dat_file_path, 'r') as f:
           print("\nFirst few lines of .dat file:")
           print("-" * 50)
           for i, line in enumerate(f):
               if i < 5:  # Show first 5 lines
                   print(f"Line {i+1}: {line.strip()}")
                   print(f"Number of columns: {len(line.strip().split())}")

       # Read with pandas to get column count
       data = pd.read_csv(dat_file_path, delim_whitespace=True, header=None)
       print("\nData shape:", data.shape)
       print("Number of columns:", len(data.columns))
       return len(data.columns)
   except Exception as e:
       print(f"Error inspecting file: {str(e)}")
       return 0

def convert_dat_to_csv(dat_file_path: str, csv_file_path: str):
   """Convert .dat sensor data to our required CSV format"""
   # First inspect the file
   n_columns = inspect_dat_file(dat_file_path)

   # Read the .dat file
   data = pd.read_csv(dat_file_path, delim_whitespace=True, header=None)

   # Define our expected columns
   expected_columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z',
                      'gyro_x', 'gyro_y', 'gyro_z', 'label']

   print("\nMapping columns...")
   if n_columns != len(expected_columns):
       print(f"Warning: Found {n_columns} columns, expected {len(expected_columns)}")
       print("Please specify column mapping:")

       # Show available columns
       print("\nAvailable columns:")
       for i in range(n_columns):
           print(f"Column {i}: Sample values: {data[i].head()}")

       # For now, let's map what we can
       mapping = {}
       for i, col in enumerate(expected_columns):
           if i < n_columns:
               mapping[i] = col
           else:
               break

       # Rename columns based on mapping
       data = data[list(mapping.keys())]
       data.columns = list(mapping.values())
   else:
       data.columns = expected_columns

   # Save as CSV
   data.to_csv(csv_file_path, index=False)
   print(f"\nConverted {dat_file_path} to {csv_file_path}")
   return data

def verify_data_format(csv_file_path: str):
   """Verify the data format matches our model requirements"""
   df = pd.read_csv(csv_file_path)

   required_columns = {'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                       'gyro_y', 'gyro_z', 'label'}
   if not required_columns.issubset(df.columns):
       print("Error: Missing required columns in CSV file.")
       return

   print("\nData verification:")
   print("Data shape:", df.shape)
   print("Column names:", list(df.columns))

   print("\nData statistics:")
   for col in df.columns:
       if df[col].dtype in [float, int]:
           print(f"Column: {col}, Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
       elif col == 'label':
           print(f"Label Distribution: {df['label'].value_counts()}")

def main():
   # Example usage
   dat_file = "D:/_Python Projects/Fun_AI/Datasets/SL-ADL1.dat"  
   csv_file = "D:/_Python Projects/Fun_AI/Datasets/A.csv"  
   try:
       data = convert_dat_to_csv(dat_file, csv_file)
       verify_data_format(csv_file)
   except Exception as e:
       print(f"Error: {str(e)}")
       import traceback
       traceback.print_exc()

if __name__ == "__main__":
   main()
