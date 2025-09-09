#!/usr/bin/env python3
"""
Medical Inventory Dataset Manager
Handles dataset creation, validation, and preparation for YOLO fine-tuning
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm
import yaml

class MedicalInventoryDataset:
    """Manages medical inventory dataset for YOLO training"""
    
    MEDICAL_CATEGORIES = {
        0: "syringe",
        1: "bandage", 
        2: "medicine_bottle",
        3: "pills_blister",
        4: "surgical_instrument",
        5: "gloves_box",
        6: "mask",
        7: "iv_bag",
        8: "thermometer",
        9: "first_aid_supply"
    }
    
    def __init__(self, dataset_root: str = "./datasets/medical_inventory"):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"
        self.splits_dir = self.dataset_root / "splits"
        
        # Create directory structure
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
    def create_dataset_yaml(self) -> None:
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': str(self.dataset_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.MEDICAL_CATEGORIES),
            'names': list(self.MEDICAL_CATEGORIES.values())
        }
        
        with open(self.dataset_root / "dataset.yaml", 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
    
    def validate_annotation(self, annotation_path: str, image_path: str) -> bool:
        """Validate YOLO format annotation"""
        try:
            # Check if image exists and get dimensions
            if not os.path.exists(image_path):
                return False
                
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Validate annotation file
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Check class ID is valid
                    if class_id not in self.MEDICAL_CATEGORIES:
                        return False
                    
                    # Check coordinates are normalized (0-1)
                    if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                        return False
                        
                except (ValueError, IndexError):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Validation error for {annotation_path}: {e}")
            return False
    
    def convert_to_yolo_format(self, bbox: Dict, img_width: int, img_height: int) -> str:
        """Convert bounding box to YOLO format"""
        x_min = bbox['x']
        y_min = bbox['y'] 
        box_width = bbox['width']
        box_height = bbox['height']
        class_id = bbox['class_id']
        
        # Convert to YOLO format (normalized center coordinates)
        x_center = (x_min + box_width / 2) / img_width
        y_center = (y_min + box_height / 2) / img_height
        norm_width = box_width / img_width
        norm_height = box_height / img_height
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Split dataset into train/val/test sets"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get all image files
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            all_images.extend(list(self.images_dir.glob(f"**/{ext}")))
        
        # Shuffle for random split
        random.shuffle(all_images)
        
        total_images = len(all_images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # Split indices
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]
        
        # Move files to appropriate directories
        splits = {
            'train': train_images,
            'val': val_images, 
            'test': test_images
        }
        
        for split_name, images in splits.items():
            for img_path in tqdm(images, desc=f"Moving {split_name} images"):
                # Move image
                dst_img = self.images_dir / split_name / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Move corresponding label file
                label_name = img_path.stem + '.txt'
                src_label = img_path.parent / label_name
                if src_label.exists():
                    dst_label = self.labels_dir / split_name / label_name
                    shutil.copy2(src_label, dst_label)
        
        # Save split information
        split_info = {
            'total_images': total_images,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
            'ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            }
        }
        
        with open(self.splits_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
    
    def generate_statistics(self) -> Dict:
        """Generate dataset statistics"""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {name: 0 for name in self.MEDICAL_CATEGORIES.values()},
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_stats = {'images': 0, 'annotations': 0, 'classes': {}}
            
            images_path = self.images_dir / split
            labels_path = self.labels_dir / split
            
            if images_path.exists():
                image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
                split_stats['images'] = len(image_files)
                stats['total_images'] += len(image_files)
                
                for img_file in image_files:
                    label_file = labels_path / (img_file.stem + '.txt')
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            annotations = f.readlines()
                        
                        split_stats['annotations'] += len(annotations)
                        stats['total_annotations'] += len(annotations)
                        
                        for ann in annotations:
                            if ann.strip():
                                class_id = int(ann.split()[0])
                                class_name = self.MEDICAL_CATEGORIES[class_id]
                                split_stats['classes'][class_name] = split_stats['classes'].get(class_name, 0) + 1
                                stats['class_distribution'][class_name] += 1
            
            stats['splits'][split] = split_stats
        
        return stats
    
    def visualize_annotations(self, split: str = 'train', num_samples: int = 5) -> None:
        """Visualize dataset annotations"""
        images_path = self.images_dir / split
        labels_path = self.labels_dir / split
        output_path = self.dataset_root / "visualizations" / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        
        for img_file in sample_files:
            # Load image
            img = cv2.imread(str(img_file))
            img_height, img_width = img.shape[:2]
            
            # Load annotations
            label_file = labels_path / (img_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
                
                for ann in annotations:
                    if ann.strip():
                        parts = ann.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Convert to pixel coordinates
                        x_center_px = int(x_center * img_width)
                        y_center_px = int(y_center * img_height)
                        width_px = int(width * img_width)
                        height_px = int(height * img_height)
                        
                        # Calculate bounding box corners
                        x1 = x_center_px - width_px // 2
                        y1 = y_center_px - height_px // 2
                        x2 = x_center_px + width_px // 2
                        y2 = y_center_px + height_px // 2
                        
                        # Draw bounding box
                        color = colors[class_id % len(colors)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        class_name = self.MEDICAL_CATEGORIES[class_id]
                        cv2.putText(img, class_name, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save visualization
            output_file = output_path / f"viz_{img_file.name}"
            cv2.imwrite(str(output_file), img)

class DataAugmentation:
    """Data augmentation for medical inventory images"""
    
    def __init__(self):
        pass
    
    def augment_dataset(self, dataset: MedicalInventoryDataset, augmentation_factor: int = 3):
        """Apply data augmentation to increase dataset size"""
        augmentations = [
            self.rotate_image,
            self.adjust_brightness,
            self.add_noise,
            self.flip_horizontal
        ]
        
        for split in ['train']:  # Only augment training data
            images_path = dataset.images_dir / split
            labels_path = dataset.labels_dir / split
            
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            
            for img_file in tqdm(image_files, desc=f"Augmenting {split} data"):
                img = cv2.imread(str(img_file))
                label_file = labels_path / (img_file.stem + '.txt')
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        annotations = f.read()
                
                # Apply augmentations
                for i in range(augmentation_factor):
                    aug_func = random.choice(augmentations)
                    aug_img = aug_func(img.copy())
                    
                    # Save augmented image and labels
                    aug_img_name = f"{img_file.stem}_aug_{i}{img_file.suffix}"
                    aug_label_name = f"{img_file.stem}_aug_{i}.txt"
                    
                    cv2.imwrite(str(images_path / aug_img_name), aug_img)
                    
                    if label_file.exists():
                        with open(labels_path / aug_label_name, 'w') as f:
                            f.write(annotations)
    
    def rotate_image(self, img: np.ndarray, max_angle: float = 15) -> np.ndarray:
        """Rotate image by random angle"""
        angle = random.uniform(-max_angle, max_angle)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
        return rotated
    
    def adjust_brightness(self, img: np.ndarray, factor_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Adjust image brightness"""
        factor = random.uniform(*factor_range)
        brightened = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        return brightened
    
    def add_noise(self, img: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_factor * 255, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        return noisy
    
    def flip_horizontal(self, img: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return cv2.flip(img, 1)

def main():
    """Example usage of dataset manager"""
    dataset = MedicalInventoryDataset()
    
    # Create dataset configuration
    dataset.create_dataset_yaml()
    
    # Generate statistics
    stats = dataset.generate_statistics()
    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Save statistics
    with open(dataset.dataset_root / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset prepared at: {dataset.dataset_root}")

if __name__ == "__main__":
    main()