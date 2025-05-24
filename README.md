# Yoga Pose Detection with PyTorch and MediaPipe

A machine learning system that detects and classifies yoga poses using PyTorch neural networks and MediaPipe pose estimation.

## 🧘 Features

- **107 Yoga Poses**: Supports classification of 95+ different yoga poses
- **Real-time Detection**: Live pose detection using webcam
- **High Accuracy**: Uses MediaPipe for precise joint tracking
- **Easy to Use**: Simple pipeline for training and inference
- **Lightweight Model**: Optimized for local deployment

## 🛠 Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd yoga-pose-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)
Open the complete pipeline in Jupyter:
```bash
jupyter notebook yoga_pose_detection_complete.ipynb
```
This notebook contains:
- Data processing and feature extraction
- Model training with advanced optimization
- Real-time inference and evaluation
- Complete documentation with markdown explanations

### Option 2: Streamlit Web App
Launch the simplified web interface:
```bash
streamlit run simple_app.py
```

### Option 3: Streamlit Cloud Web App
Access the public app hosted at:
[detectpose.streamlit.app](https://detectpose.streamlit.app/)

This provides:
- Image upload for pose detection
- Real-time camera mode
- Model information and pose library
- Dark theme with professional UI

## 📈 Performance

- **Accuracy**: ~85-95% on validation set (depends on dataset quality)
- **Speed**: Real-time detection
- **Size**: Lightweight model (~1-5MB)

## 🔧 Troubleshooting

### Common Issues

1. **Camera not working**
   ```bash
   # Try different camera ID
   python inference.py --mode camera --camera 1
   ```

2. **Python version mismatch**
   ```bash
   The model runs on Python version 3.12 or lower as mediapipe doesn't support the latest 3.13+
   ```

3. **Poor pose detection**
   - Ensure good lighting
   - Stand in full view of camera
   - Wear contrasting clothing

### Requirements Issues
If you get import errors:
```bash
pip install --upgrade torch torchvision mediapipe opencv-python
```

## 🎨 Customization

### Adding New Poses
1. Create new folder in `data/` with pose name
2. Add images to the folder
3. Re-run the training pipeline

### Model Tuning
- Increase epochs for better accuracy
- Adjust learning rate for faster convergence

## 🤝 Contributing

Feel free to:
- Add more yoga poses to the dataset
- Improve model architecture
- Optimize performance
- Fix bugs and add features

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🙏 Acknowledgments

- **MediaPipe**: For excellent pose estimation
- **PyTorch**: For deep learning framework
- **OpenCV**: For image processing
- **Yoga Community**: For pose knowledge and datasets on Kaggle - [Yoga-Poses](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset)
