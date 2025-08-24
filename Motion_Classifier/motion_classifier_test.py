import torch
import numpy as np
from motion_classifier import MotionClassifier

def test_model():
    sample_data =  torch.randn(1, 6, 100)  # Example input tensor


    model = MotionClassifier()
    model.load_state_dict(torch.load('motion_classifier_finetuned.pth'))
    model.eval()


    with torch.no_grad():
        output = model(sample_data)
        print("Model output:", output)
    
    classes = ['normal', 'panic', 'immobile']
    predicted_class = classes[output.argmax().item()]
    for i in range(10):
        print(f"Predicted class: {predicted_class}")
        print(f"Raw ouput scores: {output}")
    return print(f"\nResults: || {predicted_class} + {output} ||")

if __name__ == "__main__":
    test_model()