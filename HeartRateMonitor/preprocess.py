# preprocess_and_save.py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEQ_LEN = 100  # sequence length for LSTM

def create_sample_dataset(num_samples=1000, seed=42):
    """Generates synthetic heart rate data with anomalies"""
    np.random.seed(seed)
    normal_hr = np.random.randint(60, 100, size=num_samples)
    
    anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
    anomalies = np.zeros_like(normal_hr)
    anomalies[anomaly_indices] = np.random.randint(120, 180, size=len(anomaly_indices))
    
    data = np.column_stack((normal_hr, anomalies))
    labels = np.zeros(num_samples)
    labels[anomaly_indices] = 1
    
    return data, labels

def preprocess_data(data, labels, seq_len=SEQ_LEN):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, 0])  # heart rate sequence
        y.append(labels[i+seq_len])     # label for next step
    X = np.array(X)
    X = X.reshape(-1, seq_len, 1)  # (num_sequences, seq_len, 1)
    y = np.array(y)
    return X, y

def save_to_csv(X, y, file_path="D:/_Python Projects/Fun_AI/Datasets/heart_rate_sequences.csv"):
    """Save sequences and labels to CSV"""
    num_sequences, seq_len, _ = X.shape
    rows = []
    for seq, label in zip(X, y):
        row = seq.flatten().tolist() + [label]
        rows.append(row)
    columns = [f"hr_{i}" for i in range(seq_len)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"Saved {num_sequences} sequences to {file_path}")

def load_from_csv(file_path="heart_rate_sequences.csv"):
    """Load sequences and labels from CSV"""
    df = pd.read_csv(file_path)
    seq_len = df.shape[1] - 1
    X = df.iloc[:, :-1].values.reshape(-1, seq_len, 1)
    y = df.iloc[:, -1].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

if __name__ == "__main__":
    data, labels = create_sample_dataset()
    X, y = preprocess_data(data, labels)
    save_to_csv(X, y)
    
    # Example: Load for PyTorch
    X_torch, y_torch = load_from_csv()
    print("X shape:", X_torch.shape)  # (num_sequences, seq_len, 1)
    print("y shape:", y_torch.shape)  # (num_sequences, 1)
