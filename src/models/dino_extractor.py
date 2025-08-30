import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

class DinoFeatureExtractor:
    """DINO-based feature extractor for medical inventory comparison"""
    
    def __init__(self, model_size: str = "vits16", device: Optional[str] = None):
        """
        Initialize DINO feature extractor
        
        Args:
            model_size: Size of DINO model ('vits16', 'vits8', 'vitb16')
            device: Device to run model on (auto-detect if None)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained DINO model
        self.model = torch.hub.load('facebookresearch/dino:main', f'dino_{model_size}')
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Extract DINO features from an image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Feature tensor of shape (1, feature_dim)
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
            
        return features
    
    def get_attention_maps(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Get self-attention maps from DINO model
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Attention maps tensor
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get attention maps from last layer
        with torch.no_grad():
            # Forward pass through model
            _ = self.model(image_tensor)
            
            # Get attention weights from the last attention layer
            attentions = self.model.get_last_selfattention(image_tensor)
            
        return attentions
    
    def compare_features(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """
        Compare two feature vectors using cosine similarity
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy for sklearn
        feat1_np = features1.cpu().numpy()
        feat2_np = features2.cpu().numpy()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(feat1_np, feat2_np)[0, 0]
        
        return float(similarity)
    
    def detect_changes(self, before_image: Union[Image.Image, np.ndarray], 
                      after_image: Union[Image.Image, np.ndarray],
                      threshold: float = 0.8) -> Dict:
        """
        Detect changes between before and after images using DINO features
        
        Args:
            before_image: Before image
            after_image: After image  
            threshold: Similarity threshold for change detection
            
        Returns:
            Dictionary with change analysis results
        """
        # Extract features from both images
        before_features = self.extract_features(before_image)
        after_features = self.extract_features(after_image)
        
        # Calculate similarity
        similarity = self.compare_features(before_features, after_features)
        
        # Get attention maps for visual analysis
        before_attention = self.get_attention_maps(before_image)
        after_attention = self.get_attention_maps(after_image)
        
        # Determine if significant change occurred
        has_changed = similarity < threshold
        change_magnitude = 1.0 - similarity
        
        results = {
            'similarity': similarity,
            'has_changed': has_changed,
            'change_magnitude': change_magnitude,
            'before_features': before_features,
            'after_features': after_features,
            'before_attention': before_attention,
            'after_attention': after_attention,
            'threshold_used': threshold
        }
        
        return results
    
    def get_feature_difference_heatmap(self, before_features: torch.Tensor, 
                                     after_features: torch.Tensor,
                                     image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Generate a heatmap showing feature differences between images
        
        Args:
            before_features: Features from before image
            after_features: Features from after image
            image_size: Target size for heatmap
            
        Returns:
            Heatmap as numpy array
        """
        # Calculate feature difference
        feature_diff = torch.abs(before_features - after_features)
        
        # Convert to numpy
        diff_np = feature_diff.cpu().numpy().squeeze()
        
        # Create a simple heatmap (this is a basic implementation)
        # In practice, you might want to map features back to spatial locations
        heatmap = np.ones(image_size) * np.mean(diff_np)
        
        # Normalize to 0-255 range
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
        
        return heatmap