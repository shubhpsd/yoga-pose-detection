import torch
import torch.nn as nn

class YogaPoseClassifier(nn.Module):
    """Simple feedforward neural network for yoga pose classification"""
    
    def __init__(self, input_size=132, num_classes=107):
        super(YogaPoseClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class ImprovedYogaPoseClassifier(nn.Module):
    """Improved model with attention mechanism for landmark importance"""
    
    def __init__(self, input_size=132, num_classes=107):
        super(ImprovedYogaPoseClassifier, self).__init__()
        
        self.num_landmarks = 33
        self.landmark_dim = 4
        
        self.landmark_processor = nn.Linear(self.landmark_dim, 16)
        
        self.attention = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_landmarks * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_landmarks, self.landmark_dim)
        landmark_features = self.landmark_processor(x)
        attention_weights = self.attention(landmark_features)
        attended_features = landmark_features * attention_weights
        flattened = attended_features.view(batch_size, -1)
        return self.classifier(flattened)

def create_model(model_type='simple', num_classes=107):
    """Create and return the specified model"""
    if model_type == 'simple':
        return YogaPoseClassifier(num_classes=num_classes)
    elif model_type == 'improved':
        return ImprovedYogaPoseClassifier(num_classes=num_classes)
    else:
        raise ValueError("model_type must be 'simple' or 'improved'")
