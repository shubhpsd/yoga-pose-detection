"""
Yoga Pose Classification Model
=============================

This module contains the neural network architecture for classifying 107 different yoga poses
from MediaPipe pose landmarks. The model uses an attention mechanism to focus on important
body landmarks for each pose.

Author: Shubham Prasad
"""

import torch
import torch.nn as nn


class YogaPoseClassifier(nn.Module):
    """
    Advanced Neural Network for Yoga Pose Classification
    
    Architecture:
    - Takes 132 features (33 landmarks × 4 coordinates each)
    - Uses attention mechanism to weight landmark importance
    - Outputs probabilities for 107 yoga pose classes
    
    Features:
    - Attention mechanism for landmark importance
    - Batch normalization for training stability
    - Dropout for regularization
    - Residual connections for better gradient flow
    """
    
    def __init__(self, input_size=132, num_classes=107):
        """
        Initialize the yoga pose classifier
        
        Args:
            input_size (int): Number of input features (default: 132)
                             33 landmarks × 4 coordinates (x, y, z, visibility)
            num_classes (int): Number of yoga pose classes (default: 107)
        """
        super(YogaPoseClassifier, self).__init__()
        
        # Store dimensions for reshaping
        self.num_landmarks = 33  # MediaPipe detects 33 body landmarks
        self.landmark_dim = 4    # Each landmark has x, y, z, visibility
        
        # Process each landmark individually to extract meaningful features
        # This allows the model to understand each body part separately
        self.landmark_processor = nn.Linear(self.landmark_dim, 16)
        
        # Attention mechanism to learn which landmarks are important for each pose
        # For example: arm positions might be more important for arm balances
        # while leg positions matter more for standing poses
        self.attention = nn.Sequential(
            nn.Linear(16, 8),      # Compress to intermediate dimension
            nn.ReLU(),             # Non-linearity for complex patterns
            nn.Linear(8, 1),       # Output single attention weight per landmark
            nn.Sigmoid()           # Sigmoid ensures weights are between 0 and 1
        )
        
        # Main classification network
        # Takes all processed landmarks and classifies the pose
        self.classifier = nn.Sequential(
            # First layer: 33 landmarks × 16 features = 528 input features
            nn.Linear(self.num_landmarks * 16, 256),
            nn.ReLU(),                    # ReLU activation for non-linearity
            nn.BatchNorm1d(256),          # Normalize for stable training
            nn.Dropout(0.3),              # Dropout to prevent overfitting
            
            # Second layer: Compress to 128 features
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            # Third layer: Further compress to 64 features
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),              # Less dropout in deeper layers
            
            # Output layer: Map to number of yoga pose classes
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights using Xavier initialization for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize network weights using Xavier uniform initialization
        This helps with training stability and convergence speed
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 132)
                             Contains flattened pose landmarks
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
                         Raw scores for each yoga pose class
        """
        batch_size = x.size(0)
        
        # Reshape input from flat vector to landmark matrix
        # From: (batch_size, 132) 
        # To:   (batch_size, 33, 4)
        # This groups coordinates by landmark for processing
        x = x.view(batch_size, self.num_landmarks, self.landmark_dim)
        
        # Process each landmark through the landmark processor
        # This creates a richer representation of each body part
        # Shape: (batch_size, 33, 16)
        landmark_features = self.landmark_processor(x)
        
        # Calculate attention weights for each landmark
        # The model learns which body parts are most important for classification
        # Shape: (batch_size, 33, 1)
        attention_weights = self.attention(landmark_features)
        
        # Apply attention weights to landmark features
        # This emphasizes important landmarks and de-emphasizes others
        # Shape: (batch_size, 33, 16)
        attended_features = landmark_features * attention_weights
        
        # Flatten attended features for the classifier
        # From: (batch_size, 33, 16)
        # To:   (batch_size, 528)
        flattened = attended_features.view(batch_size, -1)
        
        # Pass through main classifier to get final predictions
        # Shape: (batch_size, num_classes)
        output = self.classifier(flattened)
        
        return output
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization
        Useful for understanding which body parts the model focuses on
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 132)
        
        Returns:
            torch.Tensor: Attention weights of shape (batch_size, 33)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_landmarks, self.landmark_dim)
        landmark_features = self.landmark_processor(x)
        attention_weights = self.attention(landmark_features)
        
        # Return flattened attention weights
        return attention_weights.squeeze(-1)  # Shape: (batch_size, 33)


def create_model(num_classes=107):
    """
    Factory function to create a yoga pose classifier
    
    Args:
        num_classes (int): Number of yoga pose classes to classify
    
    Returns:
        YogaPoseClassifier: Initialized model ready for training or inference
    """
    return YogaPoseClassifier(num_classes=num_classes)


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model
    Useful for understanding model complexity
    
    Args:
        model (torch.nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage and model information
    model = create_model()
    
    print("Yoga Pose Classifier Model Summary")
    print("=" * 50)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Input size: {132} features (33 landmarks × 4 coordinates)")
    print(f"Output size: {107} yoga pose classes")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: ~{count_parameters(model) * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 132)  # Batch size of 1
    with torch.no_grad():
        output = model(dummy_input)
        attention_weights = model.get_attention_weights(dummy_input)
    
    print(f"\nTest Results:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")