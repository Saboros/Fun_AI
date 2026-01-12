"""
Simple Real-Time Drowning Detection Monitor for Jetson Nano
Reads sensors and continuously predicts drowning risk

Usage:
    python jetson_realtime_monitor.py
"""

import torch
import numpy as np
import serial
import time
from collections import deque
import os
import sys
import datetime
import threading

sys.path.insert(0, os.path.dirname(__file__))

from model import MultiModalFusionClassifier
from fusion_preproces import FusionDataProcessor


class SimpleDrowningMonitor:
    def __init__(self, imu_port='ttyUSB0', hr_port='ttyUSB1'):
        """Simple monitor - just reads sensors and predicts"""
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = MultiModalFusionClassifier().to(self.device)
        self.model.load_state_dict(torch.load('../Models/fusion_model.pth', map_location=self.device))
        self.model.eval()
        print("Model loaded!")
        
        # Setup preprocessor
        self.preprocessor = FusionDataProcessor()
        
        # Class names
        self.classes = ['EMERGENCY', 'WARNING', 'ALERT', 'NORMAL']
        
        # Connect sensors
        self.imu_port = f'/dev/{imu_port}' if not imu_port.startswith('/') else imu_port
        self.hr_port = f'/dev/{hr_port}' if not hr_port.startswith('/') else hr_port
        
        print(f"\nConnecting to sensors...")
        self.imu = serial.Serial(self.imu_port, 115200, timeout=0.1)
        self.hr = serial.Serial(self.hr_port, 115200, timeout=0.1)
        time.sleep(2)
        print("Sensors connected!")
        
        # Data buffers
        self.motion_buffer = deque(maxlen=100)
        self.hr_buffer = deque(maxlen=100)
        
        # Current values
        self.current_hr = 0
        self.current_spo2 = 0
        
    def read_imu(self):
        """Read one IMU sample: ax,ay,az,gx,gy,gz"""
        if self.imu.in_waiting > 0:
            line = self.imu.readline().decode('utf-8', errors='ignore').strip()
            try:
                values = [float(x) for x in line.split(',')]
                if len(values) == 6:
                    return np.array(values, dtype=np.float32)
            except:
                pass
        return None
    
    def read_hr(self):
        """Read HR and SpO2: hr,spo2"""
        if self.hr.in_waiting > 0:
            line = self.hr.readline().decode('utf-8', errors='ignore').strip()
            try:
                line = line.replace('HR:', '').replace('SPO2:', '')
                values = line.replace(' ', ',').split(',')
                hr = float(values[0])
                spo2 = float(values[1])
                if 30 <= hr <= 220 and 50 <= spo2 <= 100:
                    return hr, spo2
            except:
                pass
        return None
    
    def predict(self):
        """Make prediction from buffers"""
        if len(self.motion_buffer) < 100 or len(self.hr_buffer) < 100:
            return None
        
        # Prepare motion data: (1, 6, 100)
        motion = np.array(list(self.motion_buffer), dtype=np.float32).T
        motion = motion[np.newaxis, :, :]
        
        # Prepare HR data: (1, 100)
        hr = np.array(list(self.hr_buffer), dtype=np.float32)
        hr = hr[np.newaxis, :]
        
        # Preprocess
        motion_tensor = self.preprocessor.preprocess_motion_data(motion).to(self.device)
        hr_tensor = self.preprocessor.preprocess_hr_data(hr).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(motion_tensor, hr_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()
        
        return prediction, confidence
    
    def display(self, prediction, confidence):
        """Simple display"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 50)
        print("DROWNING DETECTION MONITOR")
        print("=" * 50)
        print(f"\nHeart Rate: {self.current_hr:.0f} bpm")
        print(f"SpO2: {self.current_spo2:.0f}%")
        print(f"Buffer: {len(self.motion_buffer)}/100 samples")
        print(f"\nPREDICTION: {self.classes[prediction]}")
        print(f"Confidence: {confidence*100:.1f}%")
        
        if prediction == 0:
            print("\n*** EMERGENCY DETECTED ***")
        elif prediction == 1:
            print("\n** WARNING: Monitor closely **")
        
        print("\nPress Ctrl+C to stop")
    
    def run(self):
        """Main monitoring loop"""
        print("\n" + "=" * 50)
        print("STARTING MONITOR")
        print("=" * 50)
        print("\nFilling buffers with sensor data...")
        
        try:
            while True:
                # Read IMU
                imu_data = self.read_imu()
                if imu_data is not None:
                    self.motion_buffer.append(imu_data)
                
                # Read HR
                hr_data = self.read_hr()
                if hr_data is not None:
                    self.current_hr, self.current_spo2 = hr_data
                    self.hr_buffer.append(self.current_hr)
                
                # Predict when ready
                result = self.predict()
                if result is not None:
                    prediction, confidence = result
                    self.display(prediction, confidence)
                
                time.sleep(0.01)  # Small delay
        
        except KeyboardInterrupt:
            print("\n\nStopped!")
        finally:
            self.imu.close()
            self.hr.close()


def main():
    print("JETSON NANO DROWNING DETECTION")
    print("=" * 50)
    
    imu_port = input("IMU port (default: ttyUSB0): ").strip() or 'ttyUSB0'
    hr_port = input("HR port (default: ttyUSB1): ").strip() or 'ttyUSB1'
    
    monitor = SimpleDrowningMonitor(imu_port, hr_port)
    monitor.run()


if __name__ == "__main__":
    main()