"""
Inference Module for Yoga Pose Classification
===========================================

This module handles real-time inference on uploaded images using the trained model.
It includes pose detection, feature extraction, and classification.

Author: Shubham Prasad
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import json
import joblib
from PIL import Image

from model import YogaPoseClassifier


class YogaPoseDetector:
    """
    Complete inference pipeline for yoga pose detection and classification

    Pipeline:
    1. Load trained model and preprocessing components
    2. Extract pose landmarks from images using MediaPipe
    3. Normalize features using saved scaler
    4. Predict yoga pose using trained neural network
    5. Return predictions with confidence scores
    """

    def __init__(
        self,
        model_path="best_model.pth",
        scaler_path="scaler.pkl",
        classes_path="../data/pose_dataset_classes.json",
    ):
        """
        Initialize the yoga pose detector

        Args:
            model_path (str): Path to trained PyTorch model
            scaler_path (str): Path to saved StandardScaler
            classes_path (str): Path to JSON file with class names
        """
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Process static images (not video)
            model_complexity=1,  # Balance between accuracy and speed
            enable_segmentation=False,  # We don't need body segmentation
            min_detection_confidence=0.5,  # Minimum confidence for pose detection
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Load model and preprocessing components
        self.load_model(model_path)
        self.load_scaler(scaler_path)
        self.load_class_names(classes_path)

        print(f"🧘 Yoga pose detector initialized on {self.device}")
        print(f"📊 Model ready for {len(self.class_names)} yoga pose classes")

    def load_model(self, model_path):
        """
        Load the trained PyTorch model from checkpoint

        Args:
            model_path (str): Path to model checkpoint file
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract model information
            num_classes = checkpoint["num_classes"]

            # Create and load model
            self.model = YogaPoseClassifier(num_classes=num_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Store model info
            self.num_classes = num_classes

            print("✅ Model loaded successfully")
            print(f"   - Classes: {num_classes}")
            print(
                f"   - Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%"
            )

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def load_scaler(self, scaler_path):
        """
        Load the StandardScaler used during training

        Args:
            scaler_path (str): Path to saved scaler file
        """
        try:
            self.scaler = joblib.load(scaler_path)
            print("✅ Feature scaler loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load scaler ({e})")
            self.scaler = None

    def load_class_names(self, classes_path):
        """
        Load yoga pose class names from JSON file

        Args:
            classes_path (str): Path to JSON file with class mappings
        """
        try:
            with open(classes_path, "r") as f:
                class_mapping = json.load(f)

            # Convert to list for easy access by index
            self.class_names = [
                class_mapping[str(i)] for i in range(len(class_mapping))
            ]

            print(f"✅ Loaded {len(self.class_names)} class names")

        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            # Create generic class names as fallback
            self.class_names = [f"Pose_{i}" for i in range(self.num_classes)]

    def extract_pose_landmarks(self, image):
        """
        Extract pose landmarks from an image using MediaPipe

        Args:
            image: Input image (PIL Image, numpy array, or file path)

        Returns:
            tuple: (landmarks_array, pose_landmarks_object, processed_image)
                  Returns (None, None, None) if no pose detected
        """
        # Convert input to numpy array if needed
        if isinstance(image, str):
            # If image is a file path, load it
            image = cv2.imread(image)
            if image is None:
                print("❌ Could not load image from path")
                return None, None, None
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # If PIL Image, convert to numpy array (PIL is RGB by default)
            image_rgb = np.array(image)
            # Ensure it has the right shape
            if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
                print(f"❌ Invalid image shape: {image_rgb.shape}")
                return None, None, None
        else:
            # Assume it's already a numpy array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR format from OpenCV, convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                print(f"❌ Invalid image format or shape: {image.shape}")
                return None, None, None

        # Process the image with MediaPipe
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                # Each landmark has x, y, z coordinates and visibility score
                landmarks.extend(
                    [
                        landmark.x,  # X coordinate (normalized 0-1)
                        landmark.y,  # Y coordinate (normalized 0-1)
                        landmark.z,  # Z coordinate (depth)
                        landmark.visibility,  # Visibility confidence (0-1)
                    ]
                )

            landmarks_array = np.array(landmarks)

            # Validate that we have the expected number of features
            if len(landmarks_array) != 132:  # 33 landmarks × 4 coordinates
                print(f"⚠️  Warning: Expected 132 features, got {len(landmarks_array)}")
                return None, None, None

            return landmarks_array, results.pose_landmarks, image_rgb

        else:
            print("⚠️  No pose landmarks detected in image")
            return None, None, None

    def predict_pose(self, landmarks, top_k=5):
        """
        Predict yoga pose from extracted landmarks

        Args:
            landmarks (np.array): Pose landmarks array of shape (132,)
            top_k (int): Number of top predictions to return

        Returns:
            list: List of (pose_name, confidence) tuples sorted by confidence
        """
        if landmarks is None or len(landmarks) != 132:
            return []

        try:
            # Preprocess landmarks
            if self.scaler is not None:
                # Apply same normalization as during training
                landmarks = landmarks.reshape(1, -1)  # Shape: (1, 132)
                landmarks = self.scaler.transform(landmarks)[0]  # Shape: (132,)

            # Convert to tensor and add batch dimension
            landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(landmarks_tensor)

                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs, dim=1)[
                    0
                ]  # Shape: (num_classes,)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(
                probabilities, min(top_k, len(self.class_names))
            )

            # Format results
            predictions = []
            for i in range(len(top_probs)):
                pose_idx = top_indices[i].item()
                confidence = top_probs[i].item()
                pose_name = self.class_names[pose_idx]

                predictions.append((pose_name, confidence))

            return predictions

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return [("Error", 0.0)]

    def draw_pose_landmarks(self, image_rgb, pose_landmarks):
        """
        Draw pose landmarks and connections on the image

        Args:
            image_rgb (np.array): RGB image
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            np.array: Image with drawn landmarks
        """
        if pose_landmarks is None:
            return image_rgb

        # Create a copy to avoid modifying original
        annotated_image = image_rgb.copy()

        # Draw landmarks and connections
        self.mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            # Landmark style (red dots)
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 0, 0),  # Red color
                thickness=2,  # Point size
                circle_radius=2,
            ),
            # Connection style (green lines)
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Green color
                thickness=2,  # Line thickness
            ),
        )

        return annotated_image

    def detect_pose_from_image(self, image, draw_landmarks=True):
        """
        Complete pipeline: detect pose and classify from image

        Args:
            image: Input image (various formats supported)
            draw_landmarks (bool): Whether to draw pose landmarks

        Returns:
            dict: Results containing predictions, confidence, and annotated image
        """
        # Extract pose landmarks
        landmarks, pose_landmarks, image_rgb = self.extract_pose_landmarks(image)

        if landmarks is None:
            return {
                "success": False,
                "error": "No pose detected in image",
                "predictions": [],
                "annotated_image": None,
            }

        # Get predictions
        predictions = self.predict_pose(landmarks, top_k=5)

        # Draw landmarks if requested
        annotated_image = None
        if draw_landmarks and image_rgb is not None:
            annotated_image = self.draw_pose_landmarks(image_rgb, pose_landmarks)

        # Format results
        result = {
            "success": True,
            "predictions": predictions,
            "annotated_image": annotated_image,
            "top_prediction": predictions[0] if predictions else None,
            "confidence": predictions[0][1] if predictions else 0.0,
        }

        return result

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            dict: Model information
        """
        return {
            "num_classes": self.num_classes,
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "class_names": self.class_names[:10] + ["..."]
            if len(self.class_names) > 10
            else self.class_names,
        }


def test_detector():
    """
    Test function to verify the detector works
    """
    print("🧪 Testing yoga pose detector...")

    try:
        # Initialize detector
        detector = YogaPoseDetector()

        # Get model info
        info = detector.get_model_info()
        print(f"Model info: {info}")

        print("✅ Detector initialized successfully!")
        print("Ready to process images!")

    except Exception as e:
        print(f"❌ Error testing detector: {e}")


if __name__ == "__main__":
    test_detector()
