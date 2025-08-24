import torch
import time
import numpy as np
from motion_alarm import MotionAlarmSystem
from mock_data import MotionDataHandler

def generate_motion_pattern(pattern_type: str) -> torch.Tensor:
    """Generate different motion patterns"""
    t = np.linspace(0, 10, 100)
    
    patterns = {
        'normal': np.sin(t) + np.random.normal(0, 0.1, (6, 100)),  # Smooth walking
        'panic': np.sin(5*t) * 2 + np.random.normal(0, 0.5, (6, 100)),  # Rapid movement
        'immobile': np.zeros((6, 100)) + np.random.normal(0, 0.05, (6, 100)),  # No movement
        'falling': np.exp(-t) * np.sin(2*t) + np.random.normal(0, 0.3, (6, 100)),  # Falling pattern
        'running': np.sin(3*t) + np.cos(2*t) + np.random.normal(0, 0.2, (6, 100))  # Running pattern
    }
    
    data = patterns.get(pattern_type, patterns['normal'])
    return torch.FloatTensor(data).unsqueeze(0)

def simulate_monitoring():
    # Initialize systems
    alarm_system = MotionAlarmSystem()
    
    print("\n🔍 Motion Monitoring System")
    print("---------------------------")
    print("Patterns: . (normal), ! (panic), _ (immobile)")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        # Simulate different scenarios
        scenarios = [
            ('normal', 5),    # Normal movement for 5 seconds
            ('running', 3),   # Running for 3 seconds
            ('panic', 2),     # Panic movement for 2 seconds
            ('normal', 3),    # Back to normal for 3 seconds
            ('falling', 1),   # Falling pattern for 1 second
            ('immobile', 4)   # Immobile for 4 seconds
        ]
        
        while True:  # Continuous monitoring
            for pattern, duration in scenarios:
                X = generate_motion_pattern(pattern)
                X = (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-7)
                
                # Process the motion data
                state, confidence = alarm_system.process_motion_data(X)
                
                # Visual indicators
                if state == 'normal':
                    print(".", end="", flush=True)
                elif state == 'panic':
                    print("\n⚠️ PANIC MOVEMENT DETECTED!")
                    print(f"Confidence: {confidence:.1%}\n")
                elif state == 'immobile':
                    print("\n🔴 IMMOBILE STATE DETECTED!")
                    print(f"Confidence: {confidence:.1%}\n")
                
                time.sleep(duration)
            
    except KeyboardInterrupt:
        print("\n\n📟 Monitoring stopped by user")
        print("System shutdown complete")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print(f"Tensor shape: {X.shape if 'X' in locals() else 'Not created'}")

if __name__ == "__main__":
    simulate_monitoring()