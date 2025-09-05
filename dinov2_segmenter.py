import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import torchvision.transforms as transforms

class DINOv2MedicalSegmenter:
    """
    Advanced DINOv2-based universal object segmenter
    Uses DINOv2 features with clustering to segment ANY objects in images
    """
    
    def __init__(self, model_size: str = "vits14", device: Optional[str] = None):
        """
        Initialize DINOv2 medical segmenter
        
        Args:
            model_size: DINOv2 model size ('vits14', 'vitb14', 'vitl14', 'vitg14')
            device: Device to run model on (auto-detect if None)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DINOv2 model from torch hub
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}')
        except:
            # Fallback to DINO v1 if v2 not available
            self.model = torch.hub.load('facebookresearch/dino:main', f'dino_{model_size.replace("v2_", "")}')
        
        self.model.to(self.device)
        self.model.eval()
        
        # Patch size for attention maps
        self.patch_size = 14
        
        # Enhanced preprocessing for universal object segmentation  
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Standard DINOv2 size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def simple_kmeans(self, features: np.ndarray, k: int, max_iters: int = 20) -> np.ndarray:
        """
        Simple K-means implementation without sklearn
        
        Args:
            features: Feature array [n_samples, n_features]
            k: Number of clusters
            max_iters: Maximum iterations
            
        Returns:
            Cluster labels for each sample
        """
        n_samples, n_features = features.shape
        
        # Initialize centroids randomly
        np.random.seed(42)
        centroids = features[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((features - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(k)])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        return labels
    
    def extract_dense_features(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Extract dense patch features from DINOv2 for segmentation
        
        Args:
            image: Input image
            
        Returns:
            Dense patch features tensor [patches, features]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get patch features using get_intermediate_layers
            features = self.model.get_intermediate_layers(image_tensor, n=1, return_class_token=False)
            
            if len(features) > 0:
                # Get patch features (exclude CLS token if present)
                patch_features = features[0]  # Shape: [batch, patches, features]
                patch_features = patch_features.squeeze(0)  # Remove batch dimension
            else:
                # Fallback: use model output directly
                output = self.model(image_tensor)
                if len(output.shape) == 2:
                    # Global features - replicate for each patch
                    num_patches = (224 // self.patch_size) ** 2
                    patch_features = output.unsqueeze(0).repeat(num_patches, 1)
                else:
                    patch_features = output.squeeze(0)
        
        return patch_features
    
    def detect_objects_clustering(self, image: Union[Image.Image, np.ndarray], 
                                 n_clusters: int = 8, min_area: int = 100) -> List[Dict]:
        """
        Detect objects using DINOv2 features and clustering
        
        Args:
            image: Input image
            n_clusters: Number of clusters for segmentation
            min_area: Minimum area for detected objects
            
        Returns:
            List of detected objects with bounding boxes
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        original_size = pil_image.size
        
        # Extract dense patch features
        patch_features = self.extract_dense_features(pil_image)
        
        # Convert to numpy for clustering
        features_np = patch_features.cpu().numpy()
        
        # Apply simple K-means clustering to group similar patches
        cluster_labels = self.simple_kmeans(features_np, n_clusters)
        
        # Create segmentation map
        patches_per_side = int(np.sqrt(len(cluster_labels)))
        if patches_per_side ** 2 != len(cluster_labels):
            patches_per_side = 16  # Default for 224x224 with patch size 14
        
        cluster_map = cluster_labels.reshape(patches_per_side, patches_per_side)
        
        # Upscale cluster map to original image size
        cluster_map_resized = cv2.resize(
            cluster_map.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        detected_objects = []
        
        # Process each cluster as potential object
        for cluster_id in range(n_clusters):
            # Create binary mask for this cluster
            binary_mask = (cluster_map_resized == cluster_id).astype(np.uint8)
            
            # Find contours for this cluster
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter small regions
                if area < min_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on cluster coherence
                cluster_patches = features_np[cluster_labels == cluster_id]
                if len(cluster_patches) > 0:
                    # Use silhouette-like score as confidence
                    intra_cluster_dist = np.mean(np.var(cluster_patches, axis=0))
                    confidence = max(0.1, 1.0 - intra_cluster_dist / 10.0)
                else:
                    confidence = 0.5
                
                detected_objects.append({
                    'id': len(detected_objects),
                    'bbox': [x, y, x+w, y+h],
                    'contour': contour.tolist(),
                    'confidence': float(confidence),
                    'area': int(area),
                    'cluster_id': int(cluster_id),
                    'type': 'object'
                })
        
        return detected_objects
    
    def segment_medical_items(self, image: Union[Image.Image, np.ndarray],
                             threshold: float = 0.6,
                             min_object_size: int = 200) -> Dict:
        """
        Segment objects in image using DINOv2 features and clustering
        
        Args:
            image: Input image
            threshold: Used to determine number of clusters (0.1-0.9 -> 15-5 clusters)
            min_object_size: Minimum object size to consider
            
        Returns:
            Dictionary with segmentation results
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            input_array = image
        else:
            pil_image = image
            input_array = np.array(image)
        
        # Convert threshold to number of clusters (lower threshold = more clusters = more sensitive)
        n_clusters = max(5, int(15 * (1.0 - threshold)))
        
        # Detect objects using clustering
        detected_objects = self.detect_objects_clustering(pil_image, n_clusters, min_object_size)
        
        # Create colorful segmentation mask
        segmentation_mask = np.zeros(input_array.shape[:2], dtype=np.uint8)
        for i, obj in enumerate(detected_objects):
            try:
                cv2.fillPoly(segmentation_mask, [np.array(obj['contour'])], obj['cluster_id'] + 1)
            except:
                # Fallback for invalid contours
                x, y, w, h = obj['bbox'][:4]
                cv2.rectangle(segmentation_mask, (x, y), (x+w, y+h), obj['cluster_id'] + 1, -1)
        
        results = {
            'detected_objects': detected_objects,
            'segmentation_mask': segmentation_mask,
            'num_objects': len(detected_objects),
            'image_size': pil_image.size,
            'threshold_used': threshold,
            'clusters_used': n_clusters
        }
        
        return results
    
    def visualize_segmentation(self, image: Union[Image.Image, np.ndarray],
                              segmentation_results: Dict,
                              show_confidence: bool = True) -> Image.Image:
        """
        Create colorful visualization with contours and bounding boxes
        
        Args:
            image: Original image
            segmentation_results: Results from segment_medical_items
            show_confidence: Whether to show confidence scores
            
        Returns:
            PIL Image with colorful segmentation overlays
        """
        if isinstance(image, np.ndarray):
            vis_image = Image.fromarray(image).convert('RGB')
        else:
            vis_image = image.convert('RGB')
        
        vis_array = np.array(vis_image)
        
        # Create colorful overlay using matplotlib colormap
        detected_objects = segmentation_results['detected_objects']
        segmentation_mask = segmentation_results['segmentation_mask']
        
        if len(detected_objects) > 0:
            # Create color overlay
            overlay = np.zeros_like(vis_array)
            
            # Get distinct colors for each cluster using predefined color palette
            n_clusters = segmentation_results.get('clusters_used', 8)
            # Predefined color palette (RGB values)
            base_colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
                [255, 192, 203], # Pink
                [128, 255, 0],  # Lime
                [255, 165, 0],  # Orange2
                [0, 128, 255],  # Sky blue
            ]
            colors = np.array(base_colors[:n_clusters])
            
            # Apply colors to each cluster region
            for obj in detected_objects:
                cluster_id = obj.get('cluster_id', 0)
                color = colors[cluster_id % len(colors)]
                
                # Create mask for this object
                mask = (segmentation_mask == (cluster_id + 1))
                overlay[mask] = color
            
            # Blend overlay with original image
            alpha = 0.4
            vis_array = (1 - alpha) * vis_array + alpha * overlay
            vis_array = vis_array.astype(np.uint8)
            
            vis_image = Image.fromarray(vis_array)
        
        # Add contours and labels
        draw = ImageDraw.Draw(vis_image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Bright colors for contours
        contour_colors = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                         '#FFA500', '#800080', '#008000', '#FF69B4']
        
        for i, obj in enumerate(detected_objects):
            color = contour_colors[i % len(contour_colors)]
            
            # Draw contour outline
            if obj.get('contour'):
                try:
                    contour_points = [(int(pt[0][0]), int(pt[0][1])) for pt in obj['contour']]
                    if len(contour_points) > 2:
                        draw.polygon(contour_points, outline=color, width=3)
                except:
                    # Fallback to bounding box
                    bbox = obj['bbox']
                    draw.rectangle(bbox, outline=color, width=3)
            else:
                bbox = obj['bbox']
                draw.rectangle(bbox, outline=color, width=3)
            
            # Add object label
            bbox = obj['bbox']
            if show_confidence:
                label = f"Obj {i+1} ({obj['confidence']:.2f})"
            else:
                label = f"Object {i+1}"
                
            # Position label
            label_y = max(5, bbox[1] - 20)
            text_bbox = draw.textbbox((bbox[0], label_y), label, font=font)
            
            # Draw background for text
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], label_y), label, fill='white', font=font)
        
        # Add summary info
        n_objects = len(detected_objects)
        n_clusters = segmentation_results.get('clusters_used', 'N/A')
        summary = f"🎯 Found {n_objects} objects using {n_clusters} clusters"
        
        text_bbox = draw.textbbox((10, 10), summary, font=font)
        draw.rectangle(text_bbox, fill='black')
        draw.text((10, 10), summary, fill='white', font=font)
        
        return vis_image
    
    def get_attention_heatmap(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Generate attention heatmap for visualization
        
        Args:
            image: Input image
            
        Returns:
            Attention heatmap as numpy array
        """
        patch_tokens, attentions = self.extract_patch_features(image)
        
        # Get attention map
        attention_map = attentions[0, :, 0, 1:].reshape(37, 37)
        
        # Interpolate to original image size
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size
            
        attention_resized = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize to 0-255
        attention_normalized = ((attention_resized - attention_resized.min()) / 
                              (attention_resized.max() - attention_resized.min()) * 255).astype(np.uint8)
        
        return attention_normalized