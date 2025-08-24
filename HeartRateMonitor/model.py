import torch
import torch.nn as nn
import torch.nn.functional as F


class HeartRateAnomalyDetector(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 64, output_dim = 1):
        super(HeartRateAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)  #LSTM stands for Long Short-Term Memory
        self.dropout = nn.Dropout(0.3) #This prevents overfitting 
        self.fc = nn.Linear(hidden_dim, output_dim) # Fully connected layer
        self.sigmoid = nn.Sigmoid() #Basically binary layer outputs

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)  # LSTM layer
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)  # Sigmoid activation for binary classification
    
def save_model(model, path = "D:/_Python Projects/Fun_AI/Models/anomaly_detector.pth"):
    """Save the model checkpoint into the models file"""
    torch.save(model.state_dict(), path)
    print(f"Model save to {path}")