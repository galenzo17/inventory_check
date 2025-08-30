import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional

def create_difference_visualization(before_image: Image.Image, 
                                  after_image: Image.Image,
                                  differences: List[Dict]) -> Image.Image:
    """Create a side-by-side visualization with highlighted differences"""
    
    # Convert PIL images to numpy arrays
    before_np = np.array(before_image)
    after_np = np.array(after_image)
    
    # Create annotated versions
    before_annotated = annotate_image(before_np, differences, 'before')
    after_annotated = annotate_image(after_np, differences, 'after')
    
    # Create side-by-side comparison
    comparison = create_side_by_side(before_annotated, after_annotated)
    
    # Add legend and title
    final_image = add_legend_and_title(comparison, differences)
    
    return Image.fromarray(final_image)

def annotate_image(image: np.ndarray, differences: List[Dict], 
                   image_type: str) -> np.ndarray:
    """Annotate an image with difference markers"""
    
    annotated = image.copy()
    
    for diff in differences:
        # Skip if this difference doesn't apply to this image
        if image_type == 'before' and diff['type'] == 'added':
            continue
        if image_type == 'after' and diff['type'] == 'missing':
            continue
        
        # Get color based on difference type
        color = get_difference_color(diff['type'])
        
        # Draw bounding box if available
        if diff.get('location') and diff['location'] is not None:
            bbox = diff['location']
            annotated = draw_bbox(annotated, bbox, color, diff['item'])
        else:
            # Add text annotation if no bbox
            annotated = add_text_annotation(annotated, diff['item'], color, image_type)
    
    return annotated

def draw_bbox(image: np.ndarray, bbox: List[float], 
              color: Tuple[int, int, int], label: str) -> np.ndarray:
    """Draw a bounding box with label on the image"""
    
    img_copy = image.copy()
    
    # Convert bbox to integers
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
    
    # Add label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_height = label_size[1] + 10
    
    # Draw filled rectangle for label background
    cv2.rectangle(img_copy, (x1, y1 - label_height), 
                 (x1 + label_size[0] + 10, y1), color, -1)
    
    # Add label text
    cv2.putText(img_copy, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_copy

def add_text_annotation(image: np.ndarray, text: str, 
                       color: Tuple[int, int, int], position: str) -> np.ndarray:
    """Add text annotation when no bounding box is available"""
    
    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    
    # Position text based on image type
    if position == 'before':
        x, y = 10, 30
    else:
        x, y = 10, height - 30
    
    # Add background for text
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img_copy, (x - 5, y - text_size[1] - 5),
                 (x + text_size[0] + 5, y + 5), color, -1)
    
    # Add text
    cv2.putText(img_copy, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_copy

def create_side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Create a side-by-side comparison image"""
    
    # Ensure both images have the same height
    height = max(before.shape[0], after.shape[0])
    
    # Resize if needed
    if before.shape[0] != height:
        scale = height / before.shape[0]
        new_width = int(before.shape[1] * scale)
        before = cv2.resize(before, (new_width, height))
    
    if after.shape[0] != height:
        scale = height / after.shape[0]
        new_width = int(after.shape[1] * scale)
        after = cv2.resize(after, (new_width, height))
    
    # Add separator
    separator = np.ones((height, 10, 3), dtype=np.uint8) * 128
    
    # Concatenate horizontally
    comparison = np.hstack([before, separator, after])
    
    return comparison

def add_legend_and_title(image: np.ndarray, differences: List[Dict]) -> np.ndarray:
    """Add legend and title to the comparison image"""
    
    height, width = image.shape[:2]
    
    # Create header with title
    header_height = 80
    header = np.ones((header_height, width, 3), dtype=np.uint8) * 240
    
    # Add title
    title = "Medical Inventory Comparison"
    cv2.putText(header, title, (width // 2 - 200, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
    
    # Add subtitle
    subtitle = f"Before (Left) | After (Right) - {len(differences)} differences detected"
    cv2.putText(header, subtitle, (width // 2 - 250, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
    
    # Create legend
    legend_height = 60
    legend = np.ones((legend_height, width, 3), dtype=np.uint8) * 240
    
    # Add legend items
    legend_items = [
        ("Missing Items", get_difference_color('missing')),
        ("Added Items", get_difference_color('added')),
        ("Modified Items", get_difference_color('modified'))
    ]
    
    x_offset = 50
    for label, color in legend_items:
        # Draw color box
        cv2.rectangle(legend, (x_offset, 20), (x_offset + 30, 40), color, -1)
        
        # Add label
        cv2.putText(legend, label, (x_offset + 40, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        x_offset += 200
    
    # Combine all parts
    final_image = np.vstack([header, image, legend])
    
    return final_image

def get_difference_color(diff_type: str) -> Tuple[int, int, int]:
    """Get color for different types of differences"""
    
    colors = {
        'missing': (0, 0, 255),      # Red for missing items
        'added': (0, 255, 0),        # Green for added items
        'modified': (255, 165, 0)    # Orange for modified items
    }
    
    return colors.get(diff_type, (128, 128, 128))

def highlight_regions(image: np.ndarray, regions: List[Dict], 
                     alpha: float = 0.3) -> np.ndarray:
    """Highlight specific regions with semi-transparent overlay"""
    
    overlay = image.copy()
    output = image.copy()
    
    for region in regions:
        if region.get('bbox'):
            x1, y1, x2, y2 = [int(coord) for coord in region['bbox']]
            color = get_difference_color(region.get('type', 'modified'))
            
            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

def create_heatmap(before_image: np.ndarray, after_image: np.ndarray) -> np.ndarray:
    """Create a heatmap showing areas of change"""
    
    # Convert to grayscale
    before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
    
    # Ensure same size
    if before_gray.shape != after_gray.shape:
        after_gray = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))
    
    # Compute absolute difference
    diff = cv2.absdiff(before_gray, after_gray)
    
    # Apply Gaussian blur for smoother heatmap
    diff_blur = cv2.GaussianBlur(diff, (21, 21), 0)
    
    # Normalize to 0-255 range
    diff_norm = cv2.normalize(diff_blur, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap