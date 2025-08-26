import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import HeartRateAnomalyDetector, save_model
import pandas as pd

SEQ_LEN = 100  # must match preprocessing

def load_data(file_path="D:/_Python Projects/Fun_AI/Datasets/heart_rate_sequences.csv"):
    """Load preprocessed sequences and labels"""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values.reshape(-1, SEQ_LEN, 1)  # (num_sequences, seq_len, 1)
    y = df.iloc[:, -1].values
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y

def train_model(model, X_train, y_train, epochs=50, lr=0.001, batch_size=32):
    """Trains the model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, labels in loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

    save_model(model)

#2nd Training Loop(Fine-Tuning)
def fine_tune_model(model, X_finetune, y_finetune, epochs=20, lr=0.0005, batch_size=16):
    """Fine-tune pre-trained model with new data"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_finetune, y_finetune)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, labels in loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tuning epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

    save_model(model)

if __name__ == "__main__":
    # Load data in 3D format
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = HeartRateAnomalyDetector()

    print("=== Initial Training ===")
    train_model(model, X_train, y_train)

    print("\n=== Fine-Tuning ===")
    fine_tune_model(model, X_test, y_test)
