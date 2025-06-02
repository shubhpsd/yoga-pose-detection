import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import json
import joblib

# Try to import the model, if it fails we'll handle it gracefully
try:
    from model import create_model
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    st.warning("Model definition not found. Running in pose detection only mode.")
    
    # Create a dummy function to prevent errors
    def create_model(*args, **kwargs):
        return None

# Set page config - simple layout
st.set_page_config(
    page_title="Yoga Pose Detection",
    page_icon="🧘",
    layout="wide"  # Using centered layout instead of wide
)

class YogaPoseDetector:
    def __init__(self, model_path="best_yoga_model.pth", scaler_path="feature_scaler.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_mapping = None
        self.pose_names = None
        self.scaler = None
        self.model_loaded = False
        self.error_message = None
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Try to load model - if it fails, we'll still allow pose detection without classification
        try:
            import os
            
            # Check if required files exist
            required_files = [model_path, 'pose_dataset_classes.json']
            if not MODEL_AVAILABLE:
                self.error_message = "Model definition (model.py) not available"
                st.warning("Model definition not found. Running in pose detection only mode.")
                return
                
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                self.error_message = f"Missing files: {', '.join(missing_files)}"
                st.warning(f"Model files not found: {', '.join(missing_files)}. Running in pose detection only mode.")
                return
            
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
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                else:
                    st.warning("Scaler file not found. Predictions may be less accurate.")
                    self.scaler = None
            except Exception as e:
                st.warning(f"Error loading scaler: {e}")
                self.scaler = None
                
            self.model_loaded = True
            st.success("Model loaded successfully!")
                
        except Exception as e:
            self.error_message = str(e)
            st.error(f"Error loading model: {e}")
            st.info("Running in pose detection only mode - landmarks will still be detected and displayed.")
            
    def is_model_available(self):
        """Check if the classification model is available"""
        return self.model_loaded and self.model is not None
    
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
        if landmarks is None:
            return []
            
        if not self.is_model_available():
            return [("Pose Classification Not Available", 0.0)]
        
        try:
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
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return [("Prediction Error", 0.0)]
    
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
def main():
    st.title("🧘 Yoga Pose Detection")
    
    # Initialize detector (only once)
    @st.cache_resource
    def get_detector():
        return YogaPoseDetector()

    try:
        detector = get_detector()
        
        # Show model status
        if detector.is_model_available():
            st.success("Full pose classification available")
            
            # Display the list of identifiable poses in an expander
            if detector.pose_names:
                with st.expander("View all identifiable yoga poses"):
                    # Create columns for better readability if the list is long
                    num_poses = len(detector.pose_names)
                    cols = st.columns(3) # Adjust number of columns as needed
                    poses_per_col = (num_poses + len(cols) - 1) // len(cols) # Ceiling division
                    
                    for i, pose_name in enumerate(detector.pose_names):
                        col_index = i // poses_per_col
                        with cols[col_index]:
                            st.write(f"- {pose_name.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ Running in pose detection only mode")
            if detector.error_message:
                st.info(f"Reason: {detector.error_message}")
        
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
                        # Draw landmarks
                        annotated_image = detector.draw_landmarks_on_image(image_rgb, pose_landmarks)
                        
                        # Display results in two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption="Original Image", use_container_width=True)
                        
                        with col2:
                            st.image(annotated_image, caption="Detected Pose", use_container_width=True)
                        
                        # Predict pose only if model is available
                        if detector.is_model_available():
                            predictions = detector.predict_pose(landmarks)
                            
                            # Show top prediction
                            if predictions and predictions[0][1] > 0:
                                st.success(f"**Detected Pose: {predictions[0][0]}** (Confidence: {predictions[0][1]:.1%})")
                                
                                # Show all top 5 predictions
                                st.write("Top 5 Predictions:")
                                for i, (pose, conf) in enumerate(predictions[:5]):
                                    if conf > 0:
                                        st.write(f"{i+1}. **{pose}**: {conf:.1%}")
                        else:
                            st.info("Pose landmarks detected successfully! Model classification not available.")
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
                        # Draw landmarks
                        annotated_image = detector.draw_landmarks_on_image(image_rgb, pose_landmarks)
                        
                        # Display results in two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption="Camera Image", use_container_width=True)
                        
                        with col2:
                            st.image(annotated_image, caption="Detected Pose", use_container_width=True)
                        
                        # Predict pose only if model is available
                        if detector.is_model_available():
                            predictions = detector.predict_pose(landmarks)
                            
                            # Show top prediction
                            if predictions and predictions[0][1] > 0:
                                st.success(f"**Detected Pose: {predictions[0][0]}** (Confidence: {predictions[0][1]:.1%})")
                                
                                # Show all top 5 predictions
                                st.write("Top 5 Predictions:")
                                for i, (pose, conf) in enumerate(predictions[:5]):
                                    if conf > 0:
                                        st.write(f"{i+1}. **{pose}**: {conf:.1%}")
                        else:
                            st.info("Pose landmarks detected successfully! Model classification not available.")
                    else:
                        st.error("No pose landmarks detected in the image.")
                        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("The app encountered an error during initialization. Please check the logs for more details.")
    
    # Add a footer to the main page mentioning the developer and GitHub repository
    st.markdown("---")
    st.markdown("**Developed by Shubham Prasad**")
    st.markdown("**GitHub Repository:** [Yoga Pose Detection](https://github.com/shubhpsd/yoga-pose-detection)")

if __name__ == "__main__":
    main()
