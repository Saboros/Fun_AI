import torch
from model import HeartRateAnomalyDetector
import numpy as np  

SEQ_LEN = 100  


model = HeartRateAnomalyDetector()
model.load_state_dict(torch.load("D:/_Python Projects/Fun_AI/Models/anomaly_detector.pth"))
model.eval()

def simulate_heart_rate(num_readings = 500):
    readings = []
    for i in range(num_readings):
        if np.random.rand() < 0.05:
            readings.append(np.random.randint(120, 180))
        else:
            readings.append(np.random.randint(60, 100))
    return readings


window = []

for t, hr in enumerate(simulate_heart_rate()):
    window.append(hr)

    if len(window) > SEQ_LEN:
        sequence = np.array(window[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        with torch.no_grad():
            output = model(sequence_tensor)
            prediction = (output >= 0.5).item()


        if prediction  == 1:
            print(f"[t={t}] Anomaly detected! HR Sequence: {window[-SEQ_LEN:]}")
        else:
            print(f"[t={t}] Normal Heart Rate")