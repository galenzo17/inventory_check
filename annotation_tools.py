#!/usr/bin/env python3
"""
Annotation Tools for Medical Inventory Dataset
Pre-labeling and annotation assistance tools
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import cv2
from ultralytics import YOLO
import gradio as gr
from pathlib import Path

class PreLabeler:
    """Pre-labeling system using existing YOLO models"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5
        
        # Medical item mapping from COCO classes
        self.medical_mapping = {
            'bottle': 'medicine_bottle',
            'cup': 'medicine_bottle', 
            'knife': 'surgical_instrument',
            'scissors': 'surgical_instrument',
            'cell phone': 'thermometer',
            'remote': 'thermometer'
        }
    
    def pre_label_image(self, image_path: str, output_path: str, confidence: float = 0.5) -> Dict:
        """Generate pre-labels for an image"""
        results = self.model(image_path, conf=confidence)
        result = results[0]
        
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        annotations = []
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Check if this is a potentially medical item
                medical_class = self.medical_mapping.get(class_name)
                if medical_class or confidence > 0.8:  # High confidence items
                    
                    # Convert to YOLO format
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Assign medical class ID (needs manual review)
                    medical_class_id = 0  # Default to syringe, needs review
                    
                    annotation = f"{medical_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    annotations.append(annotation)
                    
                    detections.append({
                        'class_name': class_name,
                        'medical_class': medical_class,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'needs_review': True
                    })
        
        # Save pre-labels
        with open(output_path, 'w') as f:
            f.writelines(annotations)
        
        return {
            'annotations_count': len(annotations),
            'detections': detections,
            'needs_review': len(detections)
        }
    
    def batch_pre_label(self, images_dir: str, output_dir: str) -> Dict:
        """Pre-label a batch of images"""
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        batch_stats = {
            'total_images': len(image_files),
            'processed': 0,
            'total_detections': 0,
            'images_with_detections': 0
        }
        
        for img_file in image_files:
            output_file = output_path / (img_file.stem + '.txt')
            result = self.pre_label_image(str(img_file), str(output_file))
            
            batch_stats['processed'] += 1
            batch_stats['total_detections'] += result['annotations_count']
            if result['annotations_count'] > 0:
                batch_stats['images_with_detections'] += 1
        
        return batch_stats

class AnnotationValidator:
    """Validates annotation quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'min_bbox_size': 0.01,  # Minimum bounding box size (normalized)
            'max_bbox_size': 0.9,   # Maximum bounding box size
            'max_objects_per_image': 50,
            'valid_class_ids': list(range(10))  # 0-9 for medical categories
        }
    
    def validate_annotation_file(self, annotation_path: str, image_path: str) -> Dict:
        """Validate a single annotation file"""
        errors = []
        warnings = []
        
        if not os.path.exists(annotation_path):
            return {'valid': False, 'errors': ['Annotation file not found'], 'warnings': []}
        
        if not os.path.exists(image_path):
            return {'valid': False, 'errors': ['Image file not found'], 'warnings': []}
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > self.validation_rules['max_objects_per_image']:
            warnings.append(f"Too many objects ({len(lines)}) in image")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: Invalid format (expected 5 values)")
                continue
            
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Validate class ID
                if class_id not in self.validation_rules['valid_class_ids']:
                    errors.append(f"Line {line_num}: Invalid class ID {class_id}")
                
                # Validate coordinates
                if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                    errors.append(f"Line {line_num}: Coordinates must be normalized (0-1)")
                
                # Validate bounding box size
                if width < self.validation_rules['min_bbox_size'] or height < self.validation_rules['min_bbox_size']:
                    warnings.append(f"Line {line_num}: Very small bounding box")
                
                if width > self.validation_rules['max_bbox_size'] or height > self.validation_rules['max_bbox_size']:
                    warnings.append(f"Line {line_num}: Very large bounding box")
                
                # Check if bbox is within image bounds
                left = x_center - width/2
                right = x_center + width/2
                top = y_center - height/2
                bottom = y_center + height/2
                
                if left < 0 or right > 1 or top < 0 or bottom > 1:
                    errors.append(f"Line {line_num}: Bounding box extends outside image")
                
            except ValueError:
                errors.append(f"Line {line_num}: Invalid numeric values")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'object_count': len([l for l in lines if l.strip()])
        }
    
    def batch_validate(self, annotations_dir: str, images_dir: str) -> Dict:
        """Validate a batch of annotations"""
        annotations_path = Path(annotations_dir)
        images_path = Path(images_dir)
        
        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'detailed_results': []
        }
        
        annotation_files = list(annotations_path.glob('*.txt'))
        
        for ann_file in annotation_files:
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_img = images_path / (ann_file.stem + ext)
                if potential_img.exists():
                    img_file = potential_img
                    break
            
            if img_file is None:
                validation_results['detailed_results'].append({
                    'file': str(ann_file),
                    'valid': False,
                    'errors': ['No corresponding image found'],
                    'warnings': []
                })
                validation_results['invalid_files'] += 1
                validation_results['total_errors'] += 1
                continue
            
            result = self.validate_annotation_file(str(ann_file), str(img_file))
            result['file'] = str(ann_file)
            validation_results['detailed_results'].append(result)
            
            validation_results['total_files'] += 1
            if result['valid']:
                validation_results['valid_files'] += 1
            else:
                validation_results['invalid_files'] += 1
            
            validation_results['total_errors'] += len(result['errors'])
            validation_results['total_warnings'] += len(result['warnings'])
        
        return validation_results

class AnnotationInterface:
    """Gradio interface for annotation review and correction"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.current_image_idx = 0
        self.image_files = []
        self.load_image_list()
    
    def load_image_list(self):
        """Load list of images to annotate"""
        images_dir = self.dataset_path / "images" / "train"
        self.image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        self.image_files.sort()
    
    def get_current_image_data(self):
        """Get current image and annotations"""
        if not self.image_files:
            return None, "No images found"
        
        img_file = self.image_files[self.current_image_idx]
        annotation_file = self.dataset_path / "labels" / "train" / (img_file.stem + '.txt')
        
        # Load image
        img = Image.open(img_file)
        
        # Load annotations if they exist
        annotations = []
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotations = [line.strip() for line in f.readlines() if line.strip()]
        
        return img, f"Image {self.current_image_idx + 1}/{len(self.image_files)}\nAnnotations: {len(annotations)}"
    
    def next_image(self):
        """Move to next image"""
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
        return self.get_current_image_data()
    
    def previous_image(self):
        """Move to previous image"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
        return self.get_current_image_data()
    
    def create_interface(self):
        """Create Gradio interface for annotation"""
        with gr.Blocks(title="Medical Inventory Annotation Tool") as interface:
            gr.Markdown("# Medical Inventory Annotation Tool")
            
            with gr.Row():
                with gr.Column():
                    image_display = gr.Image(label="Current Image", height=500)
                    
                    with gr.Row():
                        prev_btn = gr.Button("← Previous")
                        next_btn = gr.Button("Next →")
                    
                    info_text = gr.Textbox(label="Info", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### Annotation Controls")
                    
                    class_selector = gr.Dropdown(
                        choices=[
                            "0: syringe",
                            "1: bandage", 
                            "2: medicine_bottle",
                            "3: pills_blister",
                            "4: surgical_instrument",
                            "5: gloves_box",
                            "6: mask",
                            "7: iv_bag",
                            "8: thermometer",
                            "9: first_aid_supply"
                        ],
                        label="Select Class",
                        value="0: syringe"
                    )
                    
                    bbox_inputs = gr.Textbox(
                        label="Bounding Box (x_center y_center width height)",
                        placeholder="0.5 0.5 0.2 0.3"
                    )
                    
                    add_annotation_btn = gr.Button("Add Annotation", variant="primary")
                    
                    annotations_display = gr.Textbox(
                        label="Current Annotations",
                        lines=10,
                        interactive=False
                    )
                    
                    save_btn = gr.Button("Save Annotations", variant="secondary")
            
            # Load initial image
            image_display.value, info_text.value = self.get_current_image_data()
            
            # Connect buttons
            prev_btn.click(
                fn=self.previous_image,
                outputs=[image_display, info_text]
            )
            
            next_btn.click(
                fn=self.next_image,
                outputs=[image_display, info_text]
            )
        
        return interface

def main():
    """Example usage of annotation tools"""
    # Pre-label some images
    pre_labeler = PreLabeler()
    
    # Validate annotations
    validator = AnnotationValidator()
    
    print("Annotation tools initialized")
    print("Use these tools to prepare your medical inventory dataset")

if __name__ == "__main__":
    main()