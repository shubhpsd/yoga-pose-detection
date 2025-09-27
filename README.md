# 🧘 Yoga Pose Detection System

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)
![Accuracy](https://img.shields.io/badge/Accuracy-74.6%25-brightgreen.svg)

A deep learning system that classifies 107 different yoga poses using computer vision and attention-based neural networks. The system combines Google's MediaPipe for pose landmark extraction with a custom neural network architecture to achieve ~75% accuracy on a challenging multi-class classification problem.

**Key Features:**

- **Real-time pose detection** with sub-100ms inference
- **Attention mechanism** that focuses on relevant body landmarks
- **Production FastAPI backend** with modern web interface
- **Advanced training pipeline** with SMOTE balancing and Standard Scaler normalization techniques
- **Complete end-to-end system** from data processing to web deployment

## System Overview

This project demonstrates a complete machine learning pipeline for computer vision:

**Problem**: Classify yoga poses from images across 107 different pose categories  
**Solution**: MediaPipe pose estimation + attention-based neural network + FastAPI deployment  
**Result**: 74.6% accuracy with real-time inference capabilities  
**Impact**: Production-ready system that handles real user uploads and provides meaningful predictions

## Quick Start: Get It Running in 2 Minutes

I've made this as simple as possible to run. Here's how to get the AI yoga detector running on your machine:

### **🚀 The Fast Way**

```bash
# 1. Clone and setup
git clone https://github.com/shubhpsd/yoga-pose-detection
cd yoga-pose-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Quick demo (works immediately)
python demo.py

# 4. Start the web interface
cd src
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# 5. Open http://localhost:8000 and upload a yoga pose image!
```

### **What You'll See**

Upload any image of a person doing yoga, and watch the algorithm analyze it in real-time:

- **Top 5 Predictions**: The most likely poses with confidence percentages
- **Pose Visualization**: See exactly which body landmarks the model is analyzing
- **Instant Results**: Sub-second inference time for real-time feedback

### **API Integration**

Want to integrate this into your own app? The FastAPI backend makes it simple:

```bash
# Test it yourself
curl -X POST "http://localhost:8000/detect-pose" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@your_yoga_photo.jpg"

# Interactive API docs at: http://localhost:8000/docs
```

### **Python SDK Usage**

```python
from src.inference import YogaPoseDetector

# Initialize once, use everywhere
detector = YogaPoseDetector()

# Analyze any image
results = detector.detect_pose_from_image("warrior_pose.jpg")

if results['success']:
    top_pose, confidence = results['predictions'][0]
    print(f"Detected: {top_pose} ({confidence:.1%} confidence)")

    # Get top 5 predictions
    for pose, conf in results['predictions']:
        print(f"{pose}: {conf:.1%}")
```

---

## 📁 **Project Structure**

```sh
yoga-pose-detection/                    # 🐐 CLEAN & PROFESSIONAL
├── 🎯 src/                            # Core ML system
│   ├── model.py                       # Neural network architecture
│   ├── train.py                       # Training pipeline with SMOTE
│   ├── inference.py                   # Real-time pose detection
│   ├── api.py                         # FastAPI web service
│   ├── best_model.pth                 # Trained model (74.6% accuracy)
│   └── scaler.pkl                     # Feature normalization
├── 🌐 web/                           # Modern web interface
│   └── index.html                     # Upload & prediction UI
├── 📚 docs/                          # Documentation & tutorials
│   └── yoga_pose_detection_tutorial.ipynb
├── 📊 data/                          # Training datasets
│   ├── pose_dataset.npz              # Processed landmarks dataset
│   ├── pose_dataset_classes.json     # Class name mappings
│   └── [107 yoga pose folders]/      # Raw training images
├── 🖼️ samples/                        # Sample test images
├── 🚀 demo.py                        # Quick start demo script
└── requirements.txt               # Production dependencies
```

---

## The Technical Deep Dive: What Makes It Work

### **The Neural Network Architecture**

After experimenting with various architectures, I settled on an attention-based approach that actually makes sense for pose detection:

- **Input Layer**: 132 features (33 MediaPipe landmarks × 4 coordinates each)
- **Attention Mechanism**: Learns which body parts matter most for each pose
- **Hidden Layers**: 256 → 128 → 64 neurons with dropout and batch normalization
- **Output**: 107 yoga pose classifications with softmax probabilities
- **Total Parameters**: 184,652 (optimized for both accuracy and speed)

### **Why Attention Matters**

The attention mechanism was a game-changer. Instead of treating all body landmarks equally, the model learns to focus on what's important:

- **Tree Pose**: Heavy attention on leg positioning and hip alignment
- **Downward Dog**: Focuses on arm angle and spine curvature
- **Warrior Poses**: Emphasizes the distinctive leg stance and arm positions

This isn't just theoretical - I can visualize which landmarks get the highest attention weights for each pose.

### **Data Imbalance Challenges**

The biggest challenge wasn't the model architecture - it was the data:

**Class Imbalance Problem**: Some poses had 200+ images, others had barely 20
**Solution**: SMOTE oversampling to balance from 5,593 → 5,992 samples

**Landmark Quality**: MediaPipe sometimes fails to detect poses in poor lighting
**Solution**: Robust preprocessing and landmark validation

**Feature Normalization**: Raw landmark coordinates are in different scales
**Solution**: Standard Scaler normalization for stable training

### **Real-World Performance**

| Metric              | Value       | What This Means                       |
| ------------------- | ----------- | ------------------------------------- |
| **Test Accuracy**   | **74.6%**   | Correctly identifies 3 out of 4 poses |
| **Model Size**      | **0.7 MB**  | Fits easily on any device             |
| **Inference Speed** | **< 100ms** | Real-time predictions                 |
| **Classes**         | **107**     | Comprehensive pose coverage           |

## Building and Extending the System

### **Training Your Own Model**

Want to retrain or improve the model? I've made it straightforward:

```bash
cd src
python train.py

# Watch the magic happen:
# - Loads 5,992 balanced training samples
# - Applies SMOTE for class balancing
# - Trains with attention mechanism and advanced optimizations
# - Saves best_model.pth when validation accuracy peaks
# - Generates training_curves.png for analysis
```

The training process typically takes 10-30 minutes on a decent GPU/CPU since we used mediapipe to reduce the computational expense. You'll see real-time metrics and the system automatically saves the best model based on validation accuracy.

### **Adding New Yoga Poses**

One of my favorite features - the system is designed to learn new poses easily:

1. **Collect Images**: Add 50+ images to `data/new_pose_name/`
2. **Update Classes**: The system automatically detects new folders
3. **Retrain**: Run `python train.py` and watch it learn your new poses
4. **Test**: The web interface immediately supports the new poses

### **Deployment Options**

**Local Development** (great for testing):

```bash
cd src
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
# Auto-reloads when you change code
```

**Production Deployment** (for real use):

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
# Multi-worker setup for handling concurrent users
```

**Docker Deployment** (coming soon):

```dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Technical Implementation Details

### **Architecture Decisions**

**MediaPipe Integration**: Google's pose detection provides reliable landmark extraction with 33 body points per image, each containing x, y, z coordinates plus visibility confidence. Reducing training resource consumption.

**Attention Mechanism**: Instead of treating all body landmarks equally, the neural network learns which landmarks matter most for each specific pose type.

**SMOTE Balancing**: Handles the natural class imbalance in yoga pose datasets by generating synthetic training examples for underrepresented poses.

**FastAPI Backend**: Provides a production-ready REST API with automatic documentation, async file handling, and proper error management.

### **Performance Optimizations**

**Model Size**: 0.7MB model file enables fast loading and deployment  
**Inference Speed**: Sub-100ms prediction time for real-time user experience  
**Memory Efficiency**: LXC container deployment keeps resource usage minimal  
**Batch Processing**: Supports multiple image analysis for batch operations

### **Future Enhancements**

**Video Analysis**: Extend to real-time video streams for live pose correction  
**Pose Sequences**: Analyze transitions between poses and flow sequences  
**Mobile Deployment**: Optimize for on-device inference using TensorFlow Lite  
**Advanced Feedback**: Provide specific adjustment recommendations beyond classification

## Contributing

Contributions are welcome! Areas for improvement:

**Dataset Expansion**: Additional yoga pose images, especially for underrepresented classes  
**Model Architecture**: Experiment with different neural network designs or attention mechanisms  
**Performance Optimization**: Improve inference speed or reduce model size  
**Web Interface**: Enhance UI/UX, add mobile responsiveness, or new features  
**Documentation**: Code examples, tutorials, or API documentation improvements

## Resources & References

**Technical Documentation**:

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/) - Deep learning framework
- [MediaPipe Documentation](https://mediapipe.dev/) - Pose estimation library
- [FastAPI Guide](https://fastapi.tiangolo.com/tutorial/) - Web API framework
- [SMOTE Paper](https://arxiv.org/abs/1106.1813) - Synthetic oversampling technique

**Dataset Source**:

- [Kaggle Yoga Poses Dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Built by [Shubham Prasad](https://github.com/shubhpsd)** | [Portfolio](https://shubhamprasad.me) | [LinkedIn](https://linkedin.com/in/shubhpsd)
