"""Models package for deepfake detection."""

from .vit_detector import ViTDeepfakeDetector, create_detector

__all__ = ['ViTDeepfakeDetector', 'create_detector']
