import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEQ_LEN = 100  # sequence length for LSTM

def normalize_spo2(spo2):
    """Normalize SpO2 from [70, 100] to [0, 1]"""
    return (spo2 - 70.0) / 30.0

def denormalize_spo2(spo2_normalized):
    """Denormalize SpO2 from [0, 1] back to [70, 100]"""
    return spo2_normalized * 30.0 + 70.0

def create_sample_dataset(num_samples=1000, seed=42):
    """Generates synthetic heart rate and SpO2 data with anomalies"""
    np.random.seed(seed)
    
    # Generate normal heart rate data
    normal_hr = np.random.randint(60, 100, size=num_samples)
    
    # Generate normal SpO2 data (95-100% for healthy individuals)
    normal_spo2 = np.random.uniform(95, 100, size=num_samples)
    
    # Create anomaly indices
    anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
    
    # Anomalous heart rates
    anomalies_hr = np.copy(normal_hr)
    anomalies_hr[anomaly_indices] = np.random.randint(120, 180, size=len(anomaly_indices))
    
    # Anomalous SpO2 (hypoxia: 70-94%)
    anomalies_spo2 = np.copy(normal_spo2)
    anomalies_spo2[anomaly_indices] = np.random.uniform(70, 94, size=len(anomaly_indices))
    
    # **NORMALIZE SpO2 to [0, 1]**
    anomalies_spo2_normalized = normalize_spo2(anomalies_spo2)
    
    # Combine data
    data = np.column_stack((anomalies_hr, anomalies_spo2))
    labels = np.zeros(num_samples)
    labels[anomaly_indices] = 1
    
    return data, labels, anomalies_spo2_normalized  # Return NORMALIZED SpO2

def preprocess_data(data, labels, spo2_values, seq_len=SEQ_LEN):
    """Create sequences for LSTM input with SpO2 targets"""
    X_hr, X_spo2, y_anomaly, y_spo2 = [], [], [], []
    
    for i in range(len(data) - seq_len):
        # Heart rate sequence
        X_hr.append(data[i:i+seq_len, 0])
        
        # SpO2 sequence (for context)
        X_spo2.append(data[i:i+seq_len, 1])
        
        # Anomaly label for next step
        y_anomaly.append(labels[i+seq_len])
        
        # SpO2 target value for next step (already normalized)
        y_spo2.append(spo2_values[i+seq_len])
    
    # Reshape for LSTM: (num_sequences, seq_len, features)
    X_hr = np.array(X_hr).reshape(-1, seq_len, 1)
    X_spo2 = np.array(X_spo2).reshape(-1, seq_len, 1)
    
    # Combine HR and SpO2 as input features (optional: use both)
    X_combined = np.concatenate([X_hr, X_spo2], axis=2)  # Shape: (N, seq_len, 2)
    
    y_anomaly = np.array(y_anomaly)
    y_spo2 = np.array(y_spo2)
    
    return X_combined, X_hr, y_anomaly, y_spo2

def save_to_csv(X_hr, X_combined, y_anomaly, y_spo2, 
                file_path="Datasets/hr_spo2_sequences.csv"):
    """Save sequences and labels to CSV"""
    num_sequences, seq_len, _ = X_hr.shape
    rows = []
    
    for i in range(num_sequences):
        # HR sequence
        hr_seq = X_hr[i].flatten().tolist()
        
        # Combined sequence (HR + SpO2)
        combined_seq = X_combined[i].flatten().tolist()
        
        # Labels (SpO2 should already be normalized [0, 1])
        row = hr_seq + [y_anomaly[i], y_spo2[i]]
        rows.append(row)
    
    columns = ([f"hr_{i}" for i in range(seq_len)] + 
               ["anomaly_label", "spo2_target"])
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"Saved {num_sequences} sequences to {file_path}")
    print(f"SpO2 range in CSV: [{df['spo2_target'].min():.2f}, {df['spo2_target'].max():.2f}]")

def load_from_csv(file_path="Datasets/hr_spo2_sequences.csv", use_spo2=False):
    """Load sequences and labels from CSV"""
    df = pd.read_csv(file_path)
    seq_len = len([col for col in df.columns if col.startswith('hr_')])
    
    # Extract HR sequences
    X_hr = df[[f"hr_{i}" for i in range(seq_len)]].values.reshape(-1, seq_len, 1)
    
    # Extract labels
    y_anomaly = df['anomaly_label'].values
    y_spo2 = df['spo2_target'].values  # Should be normalized [0, 1]
    
    X_hr_torch = torch.tensor(X_hr, dtype=torch.float32)
    y_anomaly_torch = torch.tensor(y_anomaly, dtype=torch.float32).unsqueeze(1)
    y_spo2_torch = torch.tensor(y_spo2, dtype=torch.float32).unsqueeze(1)
    
    return X_hr_torch, y_anomaly_torch, y_spo2_torch

if __name__ == "__main__":
    print("=== Generating HR + SpO2 Dataset ===")
    data, labels, spo2_values = create_sample_dataset(num_samples=1000)
    print(f"SpO2 range (normalized): [{spo2_values.min():.2f}, {spo2_values.max():.2f}]")
    
    print("\n=== Preprocessing Data ===")
    X_combined, X_hr, y_anomaly, y_spo2 = preprocess_data(data, labels, spo2_values)
    
    print(f"X_hr shape: {X_hr.shape}")  # (N, 100, 1)
    print(f"X_combined shape: {X_combined.shape}")  # (N, 100, 2)
    print(f"y_anomaly shape: {y_anomaly.shape}")
    print(f"y_spo2 shape: {y_spo2.shape}")
    print(f"y_spo2 range: [{y_spo2.min():.2f}, {y_spo2.max():.2f}]")
    
    print("\n=== Saving to CSV ===")
    save_to_csv(X_hr, X_combined, y_anomaly, y_spo2)
    
    print("\n=== Loading from CSV ===")
    X_loaded, y_anomaly_loaded, y_spo2_loaded = load_from_csv()
    print(f"Loaded X shape: {X_loaded.shape}")
    print(f"Loaded anomaly labels shape: {y_anomaly_loaded.shape}")
    print(f"Loaded SpO2 targets shape: {y_spo2_loaded.shape}")
    print(f"Loaded SpO2 range: [{y_spo2_loaded.min():.2f}, {y_spo2_loaded.max():.2f}]")
    print("\n✅ Preprocessing complete!")