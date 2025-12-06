"""Image preprocessing utilities for deepfake detection.

This module provides utilities for preprocessing images before
feeding them to the detection model.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import warnings

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    warnings.warn("MTCNN not available. Face detection will be disabled.")


class FaceDetector:
    """Face detector for cropping faces from images."""
    
    def __init__(self):
        """Initialize face detector."""
        if not MTCNN_AVAILABLE:
            raise ImportError("MTCNN is required for face detection. Install with: pip install mtcnn")
        
        self.detector = MTCNN()
    
    def detect_face(self, image: Image.Image, 
                   return_largest: bool = True) -> Optional[Image.Image]:
        """Detect and crop face from image.
        
        Args:
            image: PIL Image
            return_largest: If True, return only the largest detected face
            
        Returns:
            Cropped face image or None if no face detected
        """
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Detect faces
        faces = self.detector.detect_faces(img_array)
        
        if not faces:
            return None
        
        # Get largest face if requested
        if return_largest:
            faces = [max(faces, key=lambda x: x['box'][2] * x['box'][3])]
        
        # Extract face region
        face = faces[0]
        x, y, w, h = face['box']
        
        # Add padding
        padding = int(w * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        # Crop face
        face_img = img_array[y:y+h, x:x+w]
        
        # Convert back to PIL
        return Image.fromarray(face_img)


def resize_image(image: Image.Image, 
                size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize image to target size.
    
    Args:
        image: PIL Image
        size: Target size (width, height)
        
    Returns:
        Resized image
    """
    return image.resize(size, Image.LANCZOS)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range.
    
    Args:
        image: Numpy array image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def enhance_image(image: Image.Image, 
                 enhance_contrast: bool = True,
                 enhance_sharpness: bool = False) -> Image.Image:
    """Enhance image quality.
    
    Args:
        image: PIL Image
        enhance_contrast: Whether to enhance contrast
        enhance_sharpness: Whether to enhance sharpness
        
    Returns:
        Enhanced image
    """
    from PIL import ImageEnhance
    
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
    
    if enhance_sharpness:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
    
    return image


def preprocess_for_detection(image: Image.Image,
                            detect_face: bool = False,
                            resize: bool = True,
                            target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Complete preprocessing pipeline for deepfake detection.
    
    Args:
        image: Input PIL Image
        detect_face: Whether to detect and crop face
        resize: Whether to resize to target size
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    processed_image = image
    
    # Detect and crop face if requested
    if detect_face and MTCNN_AVAILABLE:
        try:
            face_detector = FaceDetector()
            face_img = face_detector.detect_face(processed_image)
            if face_img is not None:
                processed_image = face_img
            else:
                print("No face detected, using full image")
        except Exception as e:
            print(f"Face detection failed: {e}. Using full image.")
    
    # Resize if requested
    if resize:
        processed_image = resize_image(processed_image, target_size)
    
    return processed_image


def load_image(image_path: str) -> Image.Image:
    """Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image
    """
    try:
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {e}")


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing utilities...")
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (512, 512), color='blue')
    
    # Test preprocessing
    processed = preprocess_for_detection(
        dummy_image,
        detect_face=False,
        resize=True
    )
    
    print(f"Original size: {dummy_image.size}")
    print(f"Processed size: {processed.size}")
    print("Test successful!")
