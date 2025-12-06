"""ViT-based Deepfake Detector Module

This module provides a simple interface to load and use Vision Transformer (ViT)
models for deepfake image detection.
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ViTDeepfakeDetector:
    """Vision Transformer-based deepfake detector.
    
    This class wraps a pre-trained ViT model and provides methods for
    detecting deepfakes in images.
    """
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 device: Optional[str] = None):
        """
        Initialize the ViT detector.
        
        Args:
            model_name: Name of the ViT model from timm library
            num_classes: Number of output classes (2 for real/fake)
            pretrained: Whether to use pretrained weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, pretrained: bool) -> nn.Module:
        """Load the ViT model architecture.
        
        Args:
            pretrained: Whether to use pretrained ImageNet weights
            
        Returns:
            Loaded model
        """
        try:
            # Load base ViT model from timm
            model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=self.num_classes
            )
            print(f"Successfully loaded {self.model_name}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_weights(self, weights_path: str):
        """Load fine-tuned weights for deepfake detection.
        
        Args:
            weights_path: Path to the model weights file
        """
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using pretrained ImageNet weights instead.")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, str]:
        """Predict if an image is real or fake.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Tuple of (probability_fake, label)
            - probability_fake: float between 0 and 1
            - label: 'Real' or 'Fake'
        """
        # Preprocess image
        tensor = self.preprocess_image(image)
        
        # Forward pass
        outputs = self.model(tensor)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Get probability of fake (class 1)
        prob_fake = probs[0, 1].item()
        
        # Determine label
        label = 'Fake' if prob_fake > 0.5 else 'Real'
        
        return prob_fake, label
    
    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """Predict for multiple images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of tuples (probability_fake, label) for each image
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


def create_detector(model_name: str = 'vit_base_patch16_224',
                   weights_path: Optional[str] = None,
                   device: Optional[str] = None) -> ViTDeepfakeDetector:
    """Factory function to create a deepfake detector.
    
    Args:
        model_name: Name of the ViT model
        weights_path: Optional path to fine-tuned weights
        device: Device to use
        
    Returns:
        Initialized detector
    """
    detector = ViTDeepfakeDetector(model_name=model_name, device=device)
    
    if weights_path:
        detector.load_weights(weights_path)
    
    return detector


if __name__ == "__main__":
    # Example usage
    print("Testing ViT Deepfake Detector...")
    
    # Create detector
    detector = create_detector()
    
    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Make prediction
    prob_fake, label = detector.predict(dummy_image)
    
    print(f"Prediction: {label}")
    print(f"Fake probability: {prob_fake:.4f}")
    print("Test successful!")
