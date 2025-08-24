import torch 
from motion_classifier import MotionClassifier
from Alarm import RangeAlarmSystem
from datetime import datetime

class MotionAlarmSystem:
    def __init__(self):
        self.model = MotionClassifier()
        self.model.load_state_dict(torch.load('motion_classifier_finetuned.pth'))
        self.model.eval()

        self.alarm = RangeAlarmSystem(data_handler = None)
        self.classes = ['normal', 'panic', 'immobile']
        self.alarm_cooldown = 60
        self.last_alarm_time = None

    def process_motion_data(self, motion_data):
        """
        Process the incoming motion data and trigger alarms if necessary.

        """
        current_time = datetime.now()

        if len(motion_data.shape) == 2:
            motion_data = motion_data.unsqueeze(0)

        if motion_data.shape[0] == 0 or motion_data.numel() == 0:
            print("Error: Empty tensor received")
            return 'error', 0.0

        with torch.no_grad():
            output = self.model(motion_data)
            predicted_class = self.classes[output.argmax(dim = 1).item()]
            confidence = torch.softmax(output, dim=1).max().item()

            if predicted_class in ['panic', 'immobile']:
                if (not self.last_alarm_time or
                    (current_time - self.last_alarm_time).seconds > self.alarm_cooldown):
                    self.last_alarm_time = current_time
                    self.alarm.trigger_alarm(predicted_class, confidence)
                    self.last_alarm_time = current_time

        return predicted_class, confidence

    def _trigger_alarm(self, motion_type, confidence):
        """ Trigger alarm with motion and type confidence """
        print("\n🚨 MOTION ALERT! 🚨")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Detected: {motion_type.upper()}")
        print(f"Confidence: {confidence:.1%}")