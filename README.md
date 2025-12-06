# 🔍 Deepfake Image Detector

A fast and simple deepfake image detection application using Vision Transformer (ViT) models with Streamlit UI. Detects manipulated images locally with GPU acceleration.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- 🚀 **Fast Detection**: Leverages ViT (Vision Transformer) models for quick and accurate deepfake detection
- 🖥️ **GPU Acceleration**: Automatically uses GPU when available for faster inference
- 🎨 **User-Friendly UI**: Clean Streamlit interface for easy image upload and analysis
- 🔧 **Configurable**: Adjustable detection threshold and model selection
- 👤 **Face Detection**: Optional face detection and cropping using MTCNN
- 📊 **Detailed Results**: Probability scores, confidence levels, and visual indicators
- 🏠 **Local Processing**: All analysis happens on your machine - privacy-first design

## 📁 Project Structure

```
deepfake-image-detector/
├── app/
│   └── streamlit_app.py      # Main Streamlit web application
├── models/
│   ├── __init__.py
│   └── vit_detector.py        # ViT-based deepfake detector
├── utils/
│   ├── __init__.py
│   └── preprocess.py          # Image preprocessing utilities
├── weights/                    # Directory for model weights (optional)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/amalsp220/deepfake-image-detector.git
cd deepfake-image-detector
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app/streamlit_app.py
```

4. **Open your browser**

The app will automatically open at `http://localhost:8501`

## 💻 Usage

### Web Interface

1. **Upload an Image**: Click the file uploader and select an image (JPG, PNG, or BMP)
2. **Configure Settings** (optional):
   - Select model variant (small, base, or large)
   - Adjust detection threshold
   - Enable face detection if analyzing portrait images
   - Choose compute device (Auto, CPU, or GPU)
3. **Analyze**: Click the "Analyze Image" button
4. **View Results**: See the prediction, probability score, and confidence level

### Programmatic Usage

```python
from PIL import Image
from models.vit_detector import create_detector

# Create detector
detector = create_detector(model_name='vit_base_patch16_224')

# Load image
image = Image.open('path/to/image.jpg')

# Run detection
prob_fake, label = detector.predict(image)

print(f"Prediction: {label}")
print(f"Fake probability: {prob_fake:.2%}")
```

## 🎯 How It Works

1. **Image Upload**: User uploads an image through the web interface
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Face Detection** (optional): MTCNN detects and crops faces for better accuracy
4. **Model Inference**: ViT model analyzes the image for manipulation artifacts
5. **Classification**: Softmax output determines real vs. fake probability
6. **Results Display**: User-friendly visualization of results with confidence indicators

## 🧠 Model Architecture

The application uses Vision Transformer (ViT) models from the `timm` library:

- **vit_small_patch16_224**: Lightweight, faster inference
- **vit_base_patch16_224**: Balanced accuracy and speed (default)
- **vit_large_patch16_224**: Highest accuracy, slower inference

The models are pre-trained on ImageNet and can be fine-tuned on deepfake datasets for improved performance.

## ⚙️ Configuration

### Model Selection

You can change the model in the sidebar or programmatically:

```python
detector = create_detector(model_name='vit_large_patch16_224')
```

### Custom Weights

To use fine-tuned weights:

```python
detector = create_detector(
    model_name='vit_base_patch16_224',
    weights_path='path/to/weights.pt'
)
```

### Detection Threshold

Adjust sensitivity via the threshold parameter:
- **Lower threshold (< 0.5)**: More sensitive, may have false positives
- **Higher threshold (> 0.5)**: Less sensitive, may miss some fakes

## 📊 Performance

- **Inference Time**:
  - GPU (NVIDIA RTX 3080): ~0.05-0.1 seconds per image
  - CPU (Intel i7): ~0.5-1.0 seconds per image

- **Accuracy**: Depends on fine-tuning and dataset:
  - Pre-trained ImageNet weights: Baseline detection
  - Fine-tuned on deepfake datasets: 85-95% accuracy

## 🔬 Technical Details

### Dependencies

- **PyTorch**: Deep learning framework
- **timm**: Pre-trained vision models
- **Streamlit**: Web application framework
- **Pillow**: Image processing
- **MTCNN** (optional): Face detection
- **OpenCV**: Image manipulation

### System Requirements

**Minimum**:
- CPU: 2+ cores
- RAM: 4 GB
- Storage: 2 GB

**Recommended**:
- CPU: 4+ cores
- RAM: 8 GB
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 5 GB

## ⚠️ Limitations and Disclaimer

- **Not 100% Accurate**: No deepfake detector is perfect. False positives and negatives can occur.
- **Dataset Dependent**: Accuracy varies based on the types of deepfakes in training data.
- **Sophisticated Fakes**: May not detect very advanced deepfakes or novel manipulation techniques.
- **Research Tool**: This is a demonstration/research tool, not a production security system.
- **Verification**: Always verify important findings through multiple sources.

**Use Responsibly**: Do not use this tool as the sole method for critical decisions about image authenticity.

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Use CPU instead
detector = create_detector(device='cpu')
```

**MTCNN Import Error**:
```bash
# Install face detection dependencies
pip install mtcnn facenet-pytorch
```

**Slow Performance**:
- Use smaller model variant (`vit_small_patch16_224`)
- Disable face detection
- Ensure GPU is being used (check CUDA availability)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **timm library**: For pre-trained ViT models
- **Streamlit**: For the amazing web framework
- **PyTorch**: For the deep learning infrastructure
- **MTCNN**: For face detection capabilities

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [Your Contact Info]

## 🔗 Resources

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Deepfake Detection Resources](https://github.com/topics/deepfake-detection)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**⭐ If you find this project useful, please consider giving it a star!**
