import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fusion_preproces import FusionDataProcessor
from motion_classifier import MotionClassifier
from heartmodel import HeartRateAnomalyDetector
from model import FusionClassifier
import warnings
warnings.filterwarnings("ignore")


class FusionAITrainer:
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.preprocessor = FusionDataProcessor()
        self.class_names = ['Emergency', 'Warning', 'Alert', 'Normal']
        
    def load_training_data(self, use_synthetic=True, beach_context=True):
        """Load training data - either synthetic or from files"""
        if use_synthetic:
            if beach_context:
                print("Generating beach-specific synthetic training data...")
                motion_data, hr_data, env_data, fusion_labels = self.preprocessor.create_beach_dataset(num_samples=2000)
                
                # Preprocess data
                motion_processed = self.preprocessor.preprocess_motion_data(motion_data)
                hr_processed = self.preprocessor.preprocess_hr_data(hr_data)
                env_processed = self.preprocessor.preprocess_environmental_data(env_data)
                
                # Create sequences
                X_motion, X_hr, y = self.preprocessor.create_fusion_sequences(
                    motion_processed, hr_processed, fusion_labels
                )
                
                # Store environmental data for context
                self.env_data = env_processed
            else:
                print("Generating synthetic training data...")
                motion_data, hr_data, fusion_labels = self.preprocessor.create_fusion_dataset(num_samples=2000)
                
                # Preprocess data
                motion_processed = self.preprocessor.preprocess_motion_data(motion_data)
                hr_processed = self.preprocessor.preprocess_hr_data(hr_data)
                
                # Create sequences
                X_motion, X_hr, y = self.preprocessor.create_fusion_sequences(
                    motion_processed, hr_processed, fusion_labels
                )
        else:
            print("Loading preprocessed data from files...")
            try:
                X_motion, X_hr, y = self.preprocessor.load_fusion_data(
                    "fusion_motion_train.csv", "fusion_hr_train.csv", "fusion_labels_train.csv"
                )
            except FileNotFoundError:
                print("Preprocessed files not found. Generating synthetic data instead...")
                return self.load_training_data(use_synthetic=True)
        
        # Split data
        X_motion_train, X_motion_test, X_hr_train, X_hr_test, y_train, y_test = train_test_split(
            X_motion, X_hr, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return (X_motion_train, X_motion_test, X_hr_train, X_hr_test, y_train, y_test)
    
    def train_individual_models(self, X_motion_train, X_hr_train, y_train, epochs=50):
        """Train individual motion and heart rate models"""
        print("=== Training Individual Models ===")
        
        # Train Motion Classifier
        print("\n1. Training Motion Classifier...")
        motion_model = MotionClassifier().to(self.device)
        motion_optimizer = optim.Adam(motion_model.parameters(), lr=0.001)
        motion_criterion = nn.CrossEntropyLoss()
        
        # Convert labels for motion (0,1,2 -> normal, panic, immobile)
        motion_labels = []
        for label in y_train:
            if label == 0:  # Emergency -> Panic
                motion_labels.append(1)
            elif label == 1:  # Warning -> Immobile
                motion_labels.append(2)
            else:  # Alert/Normal -> Normal
                motion_labels.append(0)
        
        motion_labels = torch.tensor(motion_labels, dtype=torch.long).to(self.device)
        
        motion_dataset = TensorDataset(X_motion_train.to(self.device), motion_labels)
        motion_loader = DataLoader(motion_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            motion_model.train()
            total_loss = 0
            for batch_motion, batch_labels in motion_loader:
                motion_optimizer.zero_grad()
                outputs = motion_model(batch_motion)
                loss = motion_criterion(outputs, batch_labels)
                loss.backward()
                motion_optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Motion Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(motion_loader):.4f}")
        
        # Train Heart Rate Detector
        print("\n2. Training Heart Rate Anomaly Detector...")
        hr_model = HeartRateAnomalyDetector().to(self.device)
        hr_optimizer = optim.Adam(hr_model.parameters(), lr=0.001)
        hr_criterion = nn.BCELoss()
        
        # Convert labels for HR (binary: normal vs high)
        hr_labels = []
        for label in y_train:
            if label == 2:  # Alert -> High HR
                hr_labels.append(1.0)
            else:  # Others -> Normal HR
                hr_labels.append(0.0)
        
        hr_labels = torch.tensor(hr_labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        hr_dataset = TensorDataset(X_hr_train.to(self.device), hr_labels)
        hr_loader = DataLoader(hr_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            hr_model.train()
            total_loss = 0
            for batch_hr, batch_labels in hr_loader:
                hr_optimizer.zero_grad()
                outputs = hr_model(batch_hr)
                loss = hr_criterion(outputs, batch_labels)
                loss.backward()
                hr_optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"HR Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(hr_loader):.4f}")
        
        return motion_model, hr_model
    
    def create_fusion_features(self, motion_model, hr_model, X_motion, X_hr):
        """Extract features from individual models for fusion"""
        motion_model.eval()
        hr_model.eval()
        
        with torch.no_grad():
            # Get motion predictions
            motion_outputs = motion_model(X_motion.to(self.device))
            motion_probs = torch.softmax(motion_outputs, dim=1)
            
            # Get HR predictions
            hr_outputs = hr_model(X_hr.to(self.device))
            hr_probs = torch.sigmoid(hr_outputs)
            
            # Combine features: [motion_probs (3) + hr_probs (1)]
            fusion_features = torch.cat([motion_probs, hr_probs], dim=1)
        
        return fusion_features
    
    def train_fusion_model(self, motion_model, hr_model, X_motion_train, X_hr_train, y_train,
                          X_motion_test, X_hr_test, y_test, epochs=100):
        """Train the complete fusion model"""
        print("\n=== Training Fusion Model ===")
        
        # Create fusion model
        fusion_model = FusionClassifier().to(self.device)
        
        # Copy trained weights to fusion model
        fusion_model.motion_classifier.load_state_dict(motion_model.state_dict())
        fusion_model.heart_rate_detector.load_state_dict(hr_model.state_dict())
        
        # Training setup
        fusion_optimizer = optim.Adam(fusion_model.fusion_layer.parameters(), lr=0.001)
        fusion_criterion = nn.CrossEntropyLoss()
        
        # Create datasets
        train_dataset = TensorDataset(X_motion_train.to(self.device), X_hr_train.to(self.device), y_train.to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        test_dataset = TensorDataset(X_motion_test.to(self.device), X_hr_test.to(self.device), y_test.to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training
            fusion_model.train()
            total_loss = 0
            for batch_motion, batch_hr, batch_labels in train_loader:
                fusion_optimizer.zero_grad()
                outputs = fusion_model(batch_motion, batch_hr)
                loss = fusion_criterion(outputs, batch_labels)
                loss.backward()
                fusion_optimizer.step()
                total_loss += loss.item()
            
            train_losses.append(total_loss / len(train_loader))
            
            # Evaluation
            if (epoch + 1) % 10 == 0:
                fusion_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_motion, batch_hr, batch_labels in test_loader:
                        outputs = fusion_model(batch_motion, batch_hr)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels).sum().item()
                
                test_accuracy = correct / total
                test_accuracies.append(test_accuracy)
                
                print(f"Fusion Epoch [{epoch+1}/{epochs}], "
                      f"Loss: {train_losses[-1]:.4f}, "
                      f"Test Acc: {test_accuracy:.4f}")
        
        return fusion_model, train_losses, test_accuracies
    
    def evaluate_fusion_model(self, fusion_model, X_motion_test, X_hr_test, y_test):
        """Evaluate the complete fusion model"""
        print("\n=== Final Evaluation ===")
        
        fusion_model.eval()
        
        # Create test dataset
        test_dataset = TensorDataset(X_motion_test.to(self.device), X_hr_test.to(self.device), y_test.to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_motion, batch_hr, batch_labels in test_loader:
                outputs = fusion_model(batch_motion, batch_hr)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Detailed classification report
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        # Get unique classes that are actually present
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names_subset = [self.class_names[i] for i in unique_classes]
        print(classification_report(y_true, y_pred, target_names=class_names_subset, labels=unique_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy, all_predictions
    
    def predict_beach_drowning(self, motion_data, hr_data, env_data=None):
        """Beach-specific drowning prediction with environmental context"""
        # Load trained fusion model
        fusion_model = FusionClassifier()
        try:
            fusion_model.load_model('Models/fusion_model.pth')
        except FileNotFoundError:
            print("Fusion model not found. Please train the model first.")
            return None, None
        
        # Preprocess data
        motion_processed = self.preprocessor.preprocess_motion_data(motion_data)
        hr_processed = self.preprocessor.preprocess_hr_data(hr_data)
        
        # Get prediction
        if env_data is not None:
            env_processed = self.preprocessor.preprocess_environmental_data(env_data)
            prediction, probs = fusion_model.predict_with_context(
                motion_processed, hr_processed, env_processed
            )
        else:
            with torch.no_grad():
                logits = fusion_model(motion_processed, hr_processed)
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1)
        
        return prediction, probs
    
    def load_trained_models(self):
        """Load pre-trained models for prediction"""
        # Load individual models
        motion_model = MotionClassifier()
        hr_model = HeartRateAnomalyDetector()
        
        # Load weights
        motion_model.load_state_dict(torch.load('Models/fusion_motion_model.pth', map_location=self.device))
        hr_model.load_state_dict(torch.load('Models/fusion_hr_model.pth', map_location=self.device))
        
        # Load fusion model
        fusion_model = FusionClassifier()
        fusion_model.load_model('Models/fusion_model.pth')
        
        motion_model.eval()
        hr_model.eval()
        fusion_model.eval()
        
        return motion_model, hr_model, fusion_model
    
    def save_models(self, motion_model, hr_model, fusion_model, save_dir="Models/"):
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(motion_model.state_dict(), f"{save_dir}fusion_motion_model.pth")
        torch.save(hr_model.state_dict(), f"{save_dir}fusion_hr_model.pth")
        fusion_model.save_model(f"{save_dir}fusion_model.pth")
        
        print(f"\nModels saved to {save_dir}")
    
    def plot_training_history(self, train_losses, test_accuracies):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Fusion Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies)
        plt.title('Fusion Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('fusion_training_history.png')
        plt.show()


def main():
    """Main training pipeline"""
    print("=== Fusion AI Training Pipeline ===")
    
    # Initialize trainer
    trainer = FusionAITrainer()
    print(f"Using device: {trainer.device}")
    
    # Load data
    X_motion_train, X_motion_test, X_hr_train, X_hr_test, y_train, y_test = trainer.load_training_data()
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"Motion data shape: {X_motion_train.shape}")
    print(f"HR data shape: {X_hr_train.shape}")
    
    # Train individual models
    motion_model, hr_model = trainer.train_individual_models(X_motion_train, X_hr_train, y_train)
    
    # Train fusion model
    fusion_model, train_losses, test_accuracies = trainer.train_fusion_model(
        motion_model, hr_model, X_motion_train, X_hr_train, y_train,
        X_motion_test, X_hr_test, y_test
    )
    
    # Evaluate complete model
    accuracy, predictions = trainer.evaluate_fusion_model(
        fusion_model, X_motion_test, X_hr_test, y_test
    )
    
    # Save models
    trainer.save_models(motion_model, hr_model, fusion_model)
    
    # Plot training history
    trainer.plot_training_history(train_losses, test_accuracies)
    
    print("\n=== Training Complete! ===")
    print(f"Final Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()