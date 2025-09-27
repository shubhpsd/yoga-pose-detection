"""
Training Pipeline for Yoga Pose Classification
=============================================

This module contains the complete training pipeline including:
- Data loading and preprocessing
- SMOTE for class balance
- Model training with advanced optimization
- Evaluation and model saving

Author: Shubham Prasad
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import YogaPoseClassifier


class YogaTrainer:
    """
    Complete trainer class for yoga pose classification

    This class handles:
    - Data loading and preprocessing
    - SMOTE for handling class imbalance
    - Model training with advanced techniques
    - Model evaluation and saving
    """

    def __init__(self, data_path="../data/pose_dataset.npz", device="auto"):
        """
        Initialize the trainer

        Args:
            data_path (str): Path to the pose dataset NPZ file
            device (str): Device to use ('cuda', 'cpu', or 'auto')
        """
        # Set up device (GPU if available, CPU otherwise)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🎯 Using device: {self.device}")

        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()

        # Training history for plotting
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def load_data(self):
        """
        Load and preprocess the yoga pose dataset

        Steps:
        1. Load NPZ file containing pose landmarks and labels
        2. Split into train/validation/test sets
        3. Apply SMOTE to balance training classes
        4. Normalize features using StandardScaler
        5. Convert to PyTorch tensors and create data loaders
        """
        print("📊 Loading yoga pose dataset...")

        # Load the NPZ file containing processed pose data
        # This file was created by extracting MediaPipe landmarks from images
        data = np.load(self.data_path)
        features = data["features"]  # Shape: (5593, 132)
        labels = data["labels"]  # Shape: (5593,)

        print(
            f"Dataset loaded: {features.shape[0]} samples, {len(np.unique(labels))} classes"
        )

        # Check class distribution before balancing
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(
            f"Class distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}"
        )

        # Step 1: Split into train/temp and test sets (80/20 split)
        # We stratify to ensure each class is represented proportionally
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,  # 20% for testing
            random_state=42,  # For reproducible results
            stratify=labels,  # Maintain class distribution
        )

        # Step 2: Split temp into train and validation (64/16 split of original)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.2,  # 20% of temp = 16% of original
            random_state=42,
            stratify=y_temp,
        )

        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Step 3: Apply SMOTE only to training data
        # SMOTE creates synthetic samples to balance class distribution
        print("🔄 Applying SMOTE to balance training classes...")

        smote = SMOTE(
            sampling_strategy="auto",  # Balance all classes to majority class size
            random_state=42,  # For reproducible synthetic samples
            k_neighbors=5,  # Number of neighbors for synthetic sample generation
        )

        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Check new class distribution
        unique_balanced, counts_balanced = np.unique(
            y_train_balanced, return_counts=True
        )
        print(
            f"After SMOTE: {len(X_train_balanced)} samples, all classes have {counts_balanced[0]} samples"
        )

        # Step 4: Normalize features using StandardScaler
        # This ensures all features have mean=0 and std=1 for stable training
        print("📏 Normalizing features...")

        X_train_normalized = self.scaler.fit_transform(X_train_balanced)
        X_val_normalized = self.scaler.transform(X_val)
        X_test_normalized = self.scaler.transform(X_test)

        # Step 5: Convert to PyTorch tensors and create data loaders
        # DataLoaders handle batching and shuffling during training

        # Convert numpy arrays to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_normalized), torch.LongTensor(y_train_balanced)
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_normalized), torch.LongTensor(y_val)
        )

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_normalized), torch.LongTensor(y_test)
        )

        # Create data loaders with appropriate batch sizes
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Small batches for stable gradients
            shuffle=True,  # Shuffle for better generalization
            num_workers=0,  # Single-threaded for compatibility
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=64,  # Larger batches for faster validation
            shuffle=False,  # No need to shuffle validation data
            num_workers=0,
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=0
        )

        # Store number of classes for model creation
        self.num_classes = len(unique_labels)

        print("✅ Data preprocessing complete!")
        return (
            X_train_normalized,
            X_val_normalized,
            X_test_normalized,
            y_train_balanced,
            y_val,
            y_test,
        )

    def create_model(self):
        """
        Create the yoga pose classification model with optimizer and scheduler

        Components:
        - YogaPoseClassifier: Our attention-based neural network
        - AdamW optimizer: Advanced optimizer with weight decay
        - CosineAnnealingWarmRestarts: Learning rate scheduler
        - CrossEntropyLoss with label smoothing: Loss function
        """
        print("🧠 Creating model and training components...")

        # Create the model
        self.model = YogaPoseClassifier(num_classes=self.num_classes)
        self.model.to(self.device)

        # Count and display model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Model created with {trainable_params:,} trainable parameters")

        # Create optimizer: AdamW with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,  # Learning rate
            weight_decay=0.01,  # L2 regularization to prevent overfitting
        )

        # Create learning rate scheduler: Cosine annealing with warm restarts
        # This cyclically adjusts learning rate for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Initial restart period
            T_mult=2,  # Multiply period by this after each restart
            eta_min=1e-6,  # Minimum learning rate
        )

        # Create loss function: CrossEntropy with label smoothing
        # Label smoothing prevents overconfident predictions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_epoch(self):
        """
        Train the model for one epoch

        Returns:
            tuple: (average_loss, accuracy) for this epoch
        """
        self.model.train()  # Set model to training mode

        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Progress bar for training
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_data, batch_labels in pbar:
            # Move data to device (GPU if available)
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Clear gradients from previous iteration
            self.optimizer.zero_grad()

            # Forward pass: get model predictions
            outputs = self.model(batch_data)

            # Calculate loss
            loss = self.criterion(outputs, batch_labels)

            # Backward pass: calculate gradients
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update model parameters
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100 * correct_predictions / total_samples:.2f}%",
                }
            )

        # Calculate average metrics for the epoch
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct_predictions / total_samples

        return avg_loss, accuracy

    def validate_epoch(self):
        """
        Validate the model on the validation set

        Returns:
            tuple: (average_loss, accuracy) for validation set
        """
        self.model.eval()  # Set model to evaluation mode

        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Disable gradient computation for faster validation
        with torch.no_grad():
            for batch_data, batch_labels in self.val_loader:
                # Move data to device
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                outputs = self.model(batch_data)

                # Calculate loss
                loss = self.criterion(outputs, batch_labels)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
                total_loss += loss.item()

        # Calculate average metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct_predictions / total_samples

        return avg_loss, accuracy

    def train(self, max_epochs=200, patience=25, target_accuracy=85.0):
        """
        Complete training loop with early stopping

        Args:
            max_epochs (int): Maximum number of training epochs
            patience (int): Number of epochs to wait without improvement
            target_accuracy (float): Target validation accuracy to reach

        Returns:
            dict: Training history and best model info
        """
        print(f"🚀 Starting training for up to {max_epochs} epochs...")
        print(f"Target accuracy: {target_accuracy}%")
        print(f"Early stopping patience: {patience} epochs")

        best_val_accuracy = 0
        patience_counter = 0

        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print("-" * 50)

            # Train for one epoch
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate_epoch()

            # Store history for plotting
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Check for improvement
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0

                # Save best model
                self.save_model("best_model.pth", epoch, val_acc)
                print(f"✅ New best model saved! Validation accuracy: {val_acc:.2f}%")

            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs")

            # Check early stopping conditions
            if patience_counter >= patience:
                print(
                    f"\n⏹️ Early stopping triggered after {patience} epochs without improvement"
                )
                break

            if val_acc >= target_accuracy:
                print(f"\n🎯 Target accuracy {target_accuracy}% reached!")
                break

        print("\n🏁 Training completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

        # Plot training curves
        self.plot_training_curves()

        return {
            "best_val_accuracy": best_val_accuracy,
            "total_epochs": epoch + 1,
            "train_history": {
                "train_losses": self.train_losses,
                "train_accuracies": self.train_accuracies,
                "val_losses": self.val_losses,
                "val_accuracies": self.val_accuracies,
            },
        }

    def save_model(self, filepath, epoch, accuracy):
        """
        Save model checkpoint with metadata

        Args:
            filepath (str): Path to save the model
            epoch (int): Current epoch number
            accuracy (float): Current validation accuracy
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_accuracy": accuracy,
            "num_classes": self.num_classes,
            "train_history": {
                "train_losses": self.train_losses,
                "train_accuracies": self.train_accuracies,
                "val_losses": self.val_losses,
                "val_accuracies": self.val_accuracies,
            },
        }

        torch.save(checkpoint, filepath)

        # Save scaler for inference
        joblib.dump(self.scaler, "scaler.pkl")

    def plot_training_curves(self):
        """
        Plot training and validation curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label="Training Loss", color="blue", alpha=0.7)
        ax1.plot(self.val_losses, label="Validation Loss", color="red", alpha=0.7)
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(
            self.train_accuracies, label="Training Accuracy", color="blue", alpha=0.7
        )
        ax2.plot(
            self.val_accuracies, label="Validation Accuracy", color="red", alpha=0.7
        )
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("📊 Training curves saved as 'training_curves.png'")

    def evaluate_test_set(self):
        """
        Final evaluation on the test set

        Returns:
            dict: Test accuracy and detailed metrics
        """
        print("🧪 Evaluating on test set...")

        # Load best model
        checkpoint = torch.load("best_model.pth")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_data)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Calculate final accuracy
        test_accuracy = accuracy_score(all_labels, all_predictions)

        print(f"\n🎯 Final Test Accuracy: {test_accuracy * 100:.2f}%")

        return {
            "test_accuracy": test_accuracy * 100,
            "predictions": all_predictions,
            "true_labels": all_labels,
        }


def main():
    """
    Main training script
    """
    print("🧘 Yoga Pose Classification Training")
    print("=" * 50)

    # Initialize trainer
    trainer = YogaTrainer()

    # Load and preprocess data
    trainer.load_data()

    # Create model
    trainer.create_model()

    # Train model
    training_results = trainer.train(max_epochs=150, patience=20, target_accuracy=80.0)

    # Evaluate on test set
    test_results = trainer.evaluate_test_set()

    # Print final summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
    print(f"Final test accuracy: {test_results['test_accuracy']:.2f}%")
    print(f"Total epochs trained: {training_results['total_epochs']}")
    print("Model saved as: best_model.pth")
    print("Scaler saved as: scaler.pkl")


if __name__ == "__main__":
    main()
