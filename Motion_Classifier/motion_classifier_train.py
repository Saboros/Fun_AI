from mock_data import MotionDataHandler
from motion_classifier import MotionClassifier
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(X, y, model, epochs = 100, batch_size = 32, learning_rate = 0.001 ):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs},], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    data_handler = MotionDataHandler()
    model = MotionClassifier()

    # Load and preprocess real data
    print("Loading data from CSV...")
    try:
        X, y = data_handler.load_csv_data('example_motion_data.csv')
        print(f"Loaded data shape: X={X.shape}, y={y.shape}")
        
        # Train the model
        print("\nStarting training...")
        train_model(X, y, model)
        
        # Save the trained model
        torch.save(model.state_dict(), "motion_classifier_finetuned.pth")
        print("\nTraining complete! Model saved as 'motion_classifier_finetuned.pth'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTrying mock data instead...")
        
        # Fallback to mock data
        X, y = data_handler.generate_mock_data()
        X, y = data_handler.preprocess_data(X, y)
        train_model(X, y, model)
        torch.save(model.state_dict(), "motion_classifier_finetuned.pth")
