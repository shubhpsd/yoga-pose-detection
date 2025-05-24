import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import json
import joblib
from model import create_model

# Set page config - simple layout
st.set_page_config(
    page_title="Yoga Pose Detection",
    page_icon="🧘",
    layout="centered"  # Using centered layout instead of wide
)

class YogaPoseDetector:
    def __init__(self, model_path="best_yoga_model.pth", scaler_path="feature_scaler.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_mapping = None
        self.pose_names = None
        self.scaler = None
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load model
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration
            num_classes = checkpoint['num_classes']
            model_type = checkpoint.get('model_type', 'simple')
            
            # Load class mapping from JSON file
            with open('pose_dataset_classes.json', 'r') as f:
                self.class_mapping = json.load(f)
            
            # Create and load model
            self.model = create_model(
                num_classes=num_classes,
                model_type=model_type
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Convert class mapping to list for quick access
            self.pose_names = [self.class_mapping[str(i)] for i in range(num_classes)]
            
            # Load scaler
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Error loading scaler: {e}")
                self.scaler = None
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def extract_landmarks(self, image):
        """Extract pose landmarks from image"""
        try:
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(landmarks), results.pose_landmarks, image_rgb
            else:
                return None, None, image_rgb
                
        except Exception as e:
            st.error(f"Error extracting landmarks: {e}")
            return None, None, None
    
    def predict_pose(self, landmarks, top_k=5):
        """Predict yoga pose from landmarks"""
        if landmarks is None or self.model is None:
            return []
        
        # Apply feature scaling if scaler is available
        if self.scaler is not None:
            landmarks = self.scaler.transform(landmarks.reshape(1, -1))[0]
        
        # Convert to tensor
        landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(landmarks_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top K predictions
            top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(self.pose_names)))
            
            predictions = []
            for i in range(len(top_probs)):
                pose_name = self.pose_names[top_indices[i].item()]
                confidence = top_probs[i].item()
                predictions.append((pose_name, confidence))
            
            return predictions
    
    def draw_landmarks_on_image(self, image_rgb, pose_landmarks):
        """Draw pose landmarks on image"""
        if pose_landmarks is None:
            return image_rgb
        
        # Create a copy to draw on
        annotated_image = image_rgb.copy()
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            annotated_image, 
            pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        return annotated_image

# Main app
st.title("Yoga Pose Detection")

# Initialize detector (only once)
@st.cache_resource
def get_detector():
    return YogaPoseDetector()

detector = get_detector()

# Simple interface with just upload and camera options
tab1, tab2 = st.tabs(["📸 Upload Image", "📹 Use Camera"])

with tab1:
    uploaded_file = st.file_uploader("Upload a yoga pose image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        # Process image
        with st.spinner("Analyzing pose..."):
            # Extract landmarks
            landmarks, pose_landmarks, image_rgb = detector.extract_landmarks(image)
            
            if landmarks is not None:
                # Predict pose
                predictions = detector.predict_pose(landmarks)
                
                # Draw landmarks
                annotated_image = detector.draw_landmarks_on_image(image_rgb, pose_landmarks)
                
                # Display results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                
                with col2:
                    st.image(annotated_image, caption="Detected Pose", use_container_width=True)
                
                # Show top prediction
                if predictions:
                    st.success(f"**Detected Pose: {predictions[0][0]}** (Confidence: {predictions[0][1]:.1%})")
                    
                    # Show all top 5 predictions
                    st.write("Top 5 Predictions:")
                    for i, (pose, conf) in enumerate(predictions[:5]):
                        st.write(f"{i+1}. **{pose}**: {conf:.1%}")
            else:
                st.error("No pose landmarks detected in the image.")

with tab2:
    st.write("Take a picture for pose detection")
    camera_input = st.camera_input("", key="camera")
    
    if camera_input is not None:
        # Process camera image
        image = Image.open(camera_input)
        
        # Process image
        with st.spinner("Analyzing pose..."):
            # Extract landmarks
            landmarks, pose_landmarks, image_rgb = detector.extract_landmarks(image)
            
            if landmarks is not None:
                # Predict pose
                predictions = detector.predict_pose(landmarks)
                
                # Draw landmarks
                annotated_image = detector.draw_landmarks_on_image(image_rgb, pose_landmarks)
                
                # Display results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Camera Image", use_container_width=True)
                
                with col2:
                    st.image(annotated_image, caption="Detected Pose", use_container_width=True)
                
                # Show top prediction
                if predictions:
                    st.success(f"**Detected Pose: {predictions[0][0]}** (Confidence: {predictions[0][1]:.1%})")
                    
                    # Show all top 5 predictions
                    st.write("Top 5 Predictions:")
                    for i, (pose, conf) in enumerate(predictions[:5]):
                        st.write(f"{i+1}. **{pose}**: {conf:.1%}")
            else:
                st.error("No pose landmarks detected in the image.")
