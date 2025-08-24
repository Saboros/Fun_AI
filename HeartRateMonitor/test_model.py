import torch
from model import HeartRateAnomalyDetector
import pandas as pd
from sklearn.metrics import classification_report

model = HeartRateAnomalyDetector()

model.load_state_dict(torch.load("D:/_Python Projects/Fun_AI/Models/anomaly_detector.pth"))
model.eval()

SEQ_LEN = 100

df = pd.read_csv("D:/_Python Projects/Fun_AI/Datasets/heart_rate_sequences.csv")

X = df.iloc[:, :-1].values.reshape(-1, SEQ_LEN, 1)
y = df.iloc[:, -1].values

X_test = torch.tensor(X[-200:], dtype = torch.float32)
y_test = torch.tensor(y[-200:], dtype = torch.float32).unsqueeze(1)

with torch.no_grad():
    outputs = model(X_test)
    predicted =(outputs >= 0.5).float()

print(classification_report(y_test.numpy(), predicted.numpy()))