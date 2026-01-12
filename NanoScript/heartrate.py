import torch
import torch.nn as nn
import torch.nn.functional as F


class HeartRateAnomalyDetector(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 64, output_dim = 1, spo2_output=False):
        super(HeartRateAnomalyDetector, self).__init__()
        self.spo2_output = spo2_output
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)  #LSTM stands for Long Short-Term Memory
        self.dropout = nn.Dropout(0.3) #This prevents overfitting 
        
        #HR Anomaly Head
        self.fc_anomaly= nn.Linear(hidden_dim, output_dim) # Fully connected layer
        self.sigmoid = nn.Sigmoid() #Basically binary layer outputs
        
        #SPO2 Regression Head
        if self.spo2_output:
            self.fc_spo2 = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )

        
    #Forward pass
    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)  # LSTM layer
        out = self.dropout(out[:, -1, :])

        #Anomaly detection output
        anomaly_out = self.fc_anomaly(out)
        anomaly_out = self.sigmoid(anomaly_out)   # Sigmoid activation for binary classification

        if self.spo2_output:
            spo2_out = self.fc_spo2(out)
            spo2_out = torch.sigmoid(spo2_out)  # Output normalized SpO2 [0, 1]
            return anomaly_out, spo2_out

        return anomaly_out
    
    def save_model(self, path = "Models/hr_spo2_model.pth"):
        """Save the model checkpoint into the models file"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path = "Models/hr_spo2_model.pth"):
        """Load the model checkpoint"""
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")