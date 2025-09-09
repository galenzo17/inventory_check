#!/usr/bin/env python3
"""
Data Quality Checker for Medical Inventory Dataset
Ensures dataset quality and consistency
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class DataQualityChecker:
    """Comprehensive data quality assessment"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.quality_report = {
            'image_quality': {},
            'annotation_quality': {},
            'dataset_balance': {},
            'recommendations': []
        }
    
    def check_image_quality(self) -> Dict:
        """Check image quality metrics"""
        quality_issues = []
        image_stats = {
            'total_images': 0,
            'resolution_stats': {'width': [], 'height': []},
            'aspect_ratios': [],
            'file_sizes': [],
            'corrupted_images': [],
            'low_quality_images': []
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / 'images' / split
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_file in image_files:
                try:
                    img = Image.open(img_file)
                    width, height = img.size
                    file_size = img_file.stat().st_size
                    
                    image_stats['total_images'] += 1
                    image_stats['resolution_stats']['width'].append(width)
                    image_stats['resolution_stats']['height'].append(height)
                    image_stats['aspect_ratios'].append(width / height)
                    image_stats['file_sizes'].append(file_size)
                    
                    # Check for very small images
                    if width < 224 or height < 224:
                        quality_issues.append(f"Small image: {img_file} ({width}x{height})")
                        image_stats['low_quality_images'].append(str(img_file))
                    
                    # Check for very large images
                    if width > 4096 or height > 4096:
                        quality_issues.append(f"Very large image: {img_file} ({width}x{height})")
                    
                    # Check image sharpness (Laplacian variance)
                    img_cv = cv2.imread(str(img_file))
                    if img_cv is not None:
                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if blur_score < 100:  # Threshold for blur detection
                            quality_issues.append(f"Blurry image: {img_file} (score: {blur_score:.2f})")
                            image_stats['low_quality_images'].append(str(img_file))
                
                except Exception as e:
                    quality_issues.append(f"Corrupted image: {img_file} - {str(e)}")
                    image_stats['corrupted_images'].append(str(img_file))
        
        # Calculate statistics
        if image_stats['resolution_stats']['width']:
            image_stats['avg_width'] = np.mean(image_stats['resolution_stats']['width'])
            image_stats['avg_height'] = np.mean(image_stats['resolution_stats']['height'])
            image_stats['avg_aspect_ratio'] = np.mean(image_stats['aspect_ratios'])
            image_stats['avg_file_size'] = np.mean(image_stats['file_sizes'])
        
        self.quality_report['image_quality'] = {
            'stats': image_stats,
            'issues': quality_issues,
            'issue_count': len(quality_issues)
        }
        
        return image_stats
    
    def check_annotation_quality(self) -> Dict:
        """Check annotation quality and consistency"""
        annotation_issues = []
        annotation_stats = {
            'total_annotations': 0,
            'annotations_per_image': [],
            'class_distribution': defaultdict(int),
            'bbox_sizes': [],
            'bbox_aspect_ratios': [],
            'missing_annotations': [],
            'invalid_annotations': []
        }
        
        medical_categories = {
            0: "syringe", 1: "bandage", 2: "medicine_bottle", 3: "pills_blister",
            4: "surgical_instrument", 5: "gloves_box", 6: "mask", 7: "iv_bag",
            8: "thermometer", 9: "first_aid_supply"
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_file in image_files:
                label_file = labels_dir / (img_file.stem + '.txt')
                
                if not label_file.exists():
                    annotation_issues.append(f"Missing annotation: {img_file}")
                    annotation_stats['missing_annotations'].append(str(img_file))
                    continue
                
                try:
                    with open(label_file, 'r') as f:
                        annotations = [line.strip() for line in f.readlines() if line.strip()]
                    
                    annotation_stats['annotations_per_image'].append(len(annotations))
                    annotation_stats['total_annotations'] += len(annotations)
                    
                    for line_num, annotation in enumerate(annotations, 1):
                        parts = annotation.split()
                        if len(parts) != 5:
                            annotation_issues.append(f"Invalid format in {label_file}, line {line_num}")
                            annotation_stats['invalid_annotations'].append(f"{label_file}:{line_num}")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Validate class ID
                            if class_id not in medical_categories:
                                annotation_issues.append(f"Invalid class ID {class_id} in {label_file}, line {line_num}")
                                continue
                            
                            # Update class distribution
                            annotation_stats['class_distribution'][medical_categories[class_id]] += 1
                            
                            # Validate coordinates
                            if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                                annotation_issues.append(f"Invalid coordinates in {label_file}, line {line_num}")
                                continue
                            
                            # Track bounding box statistics
                            annotation_stats['bbox_sizes'].append(width * height)
                            if height > 0:
                                annotation_stats['bbox_aspect_ratios'].append(width / height)
                            
                            # Check for very small bounding boxes
                            if width < 0.01 or height < 0.01:
                                annotation_issues.append(f"Very small bbox in {label_file}, line {line_num}")
                            
                        except ValueError:
                            annotation_issues.append(f"Invalid numeric values in {label_file}, line {line_num}")
                            annotation_stats['invalid_annotations'].append(f"{label_file}:{line_num}")
                
                except Exception as e:
                    annotation_issues.append(f"Error reading {label_file}: {str(e)}")
        
        # Calculate statistics
        if annotation_stats['annotations_per_image']:
            annotation_stats['avg_annotations_per_image'] = np.mean(annotation_stats['annotations_per_image'])
            annotation_stats['median_annotations_per_image'] = np.median(annotation_stats['annotations_per_image'])
        
        if annotation_stats['bbox_sizes']:
            annotation_stats['avg_bbox_size'] = np.mean(annotation_stats['bbox_sizes'])
            annotation_stats['median_bbox_size'] = np.median(annotation_stats['bbox_sizes'])
        
        self.quality_report['annotation_quality'] = {
            'stats': annotation_stats,
            'issues': annotation_issues,
            'issue_count': len(annotation_issues)
        }
        
        return annotation_stats
    
    def check_dataset_balance(self) -> Dict:
        """Check class balance and distribution"""
        balance_stats = {
            'class_counts': {},
            'split_distribution': {},
            'imbalance_ratio': 0,
            'recommendations': []
        }
        
        total_class_counts = Counter()
        split_stats = {}
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_path / 'labels' / split
            if not labels_dir.exists():
                continue
            
            split_class_counts = Counter()
            label_files = list(labels_dir.glob('*.txt'))
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        annotations = [line.strip() for line in f.readlines() if line.strip()]
                    
                    for annotation in annotations:
                        parts = annotation.split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(parts[0])
                                split_class_counts[class_id] += 1
                                total_class_counts[class_id] += 1
                            except ValueError:
                                continue
                
                except Exception:
                    continue
            
            split_stats[split] = dict(split_class_counts)
        
        balance_stats['class_counts'] = dict(total_class_counts)
        balance_stats['split_distribution'] = split_stats
        
        # Calculate imbalance ratio
        if total_class_counts:
            max_count = max(total_class_counts.values())
            min_count = min(total_class_counts.values())
            balance_stats['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')
            
            # Generate recommendations
            if balance_stats['imbalance_ratio'] > 10:
                balance_stats['recommendations'].append("High class imbalance detected. Consider data augmentation for underrepresented classes.")
            
            if min_count < 50:
                balance_stats['recommendations'].append("Some classes have very few examples. Consider collecting more data.")
        
        self.quality_report['dataset_balance'] = balance_stats
        return balance_stats
    
    def generate_visualizations(self) -> None:
        """Generate quality report visualizations"""
        viz_dir = self.dataset_path / 'quality_report'
        viz_dir.mkdir(exist_ok=True)
        
        # Class distribution plot
        if 'dataset_balance' in self.quality_report:
            class_counts = self.quality_report['dataset_balance']['class_counts']
            if class_counts:
                medical_categories = {
                    0: "syringe", 1: "bandage", 2: "medicine_bottle", 3: "pills_blister",
                    4: "surgical_instrument", 5: "gloves_box", 6: "mask", 7: "iv_bag",
                    8: "thermometer", 9: "first_aid_supply"
                }
                
                labels = [medical_categories.get(k, f"Class {k}") for k in class_counts.keys()]
                counts = list(class_counts.values())
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(labels, counts)
                plt.title('Class Distribution in Dataset')
                plt.xlabel('Medical Item Classes')
                plt.ylabel('Number of Annotations')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Image resolution distribution
        if 'image_quality' in self.quality_report:
            width_data = self.quality_report['image_quality']['stats']['resolution_stats']['width']
            height_data = self.quality_report['image_quality']['stats']['resolution_stats']['height']
            
            if width_data and height_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.hist(width_data, bins=20, alpha=0.7, color='blue')
                ax1.set_title('Image Width Distribution')
                ax1.set_xlabel('Width (pixels)')
                ax1.set_ylabel('Frequency')
                
                ax2.hist(height_data, bins=20, alpha=0.7, color='green')
                ax2.set_title('Image Height Distribution')
                ax2.set_xlabel('Height (pixels)')
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'resolution_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality report"""
        print("Checking image quality...")
        self.check_image_quality()
        
        print("Checking annotation quality...")
        self.check_annotation_quality()
        
        print("Checking dataset balance...")
        self.check_dataset_balance()
        
        print("Generating visualizations...")
        self.generate_visualizations()
        
        # Compile overall recommendations
        overall_recommendations = []
        
        # Image quality recommendations
        if self.quality_report['image_quality']['issue_count'] > 0:
            overall_recommendations.append("Address image quality issues before training")
        
        # Annotation quality recommendations
        if self.quality_report['annotation_quality']['issue_count'] > 0:
            overall_recommendations.append("Fix annotation errors and inconsistencies")
        
        # Balance recommendations
        overall_recommendations.extend(self.quality_report['dataset_balance']['recommendations'])
        
        self.quality_report['recommendations'] = overall_recommendations
        
        # Save report
        report_path = self.dataset_path / 'quality_report' / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        
        print(f"Quality report saved to: {report_path}")
        return self.quality_report

def main():
    """Example usage"""
    dataset_path = "./datasets/medical_inventory"
    
    checker = DataQualityChecker(dataset_path)
    report = checker.generate_quality_report()
    
    print("\n=== Dataset Quality Summary ===")
    print(f"Total images: {report['image_quality']['stats']['total_images']}")
    print(f"Total annotations: {report['annotation_quality']['stats']['total_annotations']}")
    print(f"Image quality issues: {report['image_quality']['issue_count']}")
    print(f"Annotation quality issues: {report['annotation_quality']['issue_count']}")
    print(f"Class imbalance ratio: {report['dataset_balance']['imbalance_ratio']:.2f}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()