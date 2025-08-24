import time
from datetime import datetime

class RangeAlarmSystem:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.alarm_triggered = False
        self.last_alarm_time = None
        self.alarm_cooldown = 60

    
    def check_and_alert(self, signal_strength: float, distance: float = None) -> None:
        current_time = datetime.now()

        if not self.data_handler.check_range(signal_strength, distance):
            if not self.alarm_active or \
            (self.last_alarm_time and (current_time - self.last_alarm_time).seconds > self.alarm_cooldown):
                self.trigger_alarm(signal_strength, distance)
                self.last_alarm_time = current_time
                self.alarm_active = True
        
        else:
            self.alarm_active = False

    def trigger_alarm(self, signal_strength: float, distance: float = None) -> None:
        print("\n🚨 ALERT: Device Out of Range! 🚨")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Signal Strength: {signal_strength} dBm")
        if distance:
            print(f"Distance: {distance:.1f}m")
        
        print("Please check the device location!")
