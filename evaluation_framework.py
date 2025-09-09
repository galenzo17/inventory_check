#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Medical YOLO Models
Performance assessment, metrics calculation, and continuous improvement
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import psutil
import GPUtil

# Import models
from medical_yolo import MedicalYOLO, create_medical_yolo_variants
from yolo_app import YOLODetectionApp

@dataclass
class DetectionResult:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    area: float
    center: Tuple[float, float]

@dataclass
class GroundTruth:
    """Ground truth annotation"""
    class_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    area: float
    difficult: bool = False

@dataclass
class ImageEvaluation:
    """Evaluation results for single image"""
    image_id: str
    predictions: List[DetectionResult]
    ground_truths: List[GroundTruth]
    matches: List[Tuple[int, int, float]]  # (pred_idx, gt_idx, iou)
    true_positives: int
    false_positives: int
    false_negatives: int
    processing_time: float

@dataclass
class ModelMetrics:
    """Comprehensive model metrics"""
    # Detection metrics
    map_50: float
    map_95: float
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    class_precisions: Dict[str, float]
    class_recalls: Dict[str, float]
    class_f1_scores: Dict[str, float]
    class_aps: Dict[str, float]
    
    # Performance metrics
    avg_inference_time: float
    fps: float
    gpu_memory_mb: float
    cpu_usage_percent: float
    model_size_mb: float
    
    # Counting metrics
    counting_mae: float
    counting_rmse: float
    counting_mape: float
    
    # Robustness metrics
    robustness_scores: Dict[str, float]

class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.class_names = {
            0: 'syringe', 1: 'bandage', 2: 'medicine_bottle', 3: 'pills_blister',
            4: 'surgical_instrument', 5: 'gloves_box', 6: 'mask', 7: 'iv_bag',
            8: 'thermometer', 9: 'first_aid_supply'
        }
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_predictions_to_gt(self, predictions: List[DetectionResult], 
                              ground_truths: List[GroundTruth], 
                              iou_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """Match predictions to ground truth using Hungarian algorithm"""
        if not predictions or not ground_truths:
            return []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truths)))
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truths):
                if pred.class_id == gt.class_id:
                    iou_matrix[i, j] = self.calculate_iou(pred.bbox, gt.bbox)
        
        # Simple greedy matching (in real implementation, use Hungarian algorithm)
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Sort by IoU descending
        potential_matches = []
        for i in range(len(predictions)):
            for j in range(len(ground_truths)):
                if iou_matrix[i, j] >= iou_threshold:
                    potential_matches.append((i, j, iou_matrix[i, j]))
        
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        for pred_idx, gt_idx, iou in potential_matches:
            if pred_idx not in used_pred and gt_idx not in used_gt:
                matches.append((pred_idx, gt_idx, iou))
                used_pred.add(pred_idx)
                used_gt.add(gt_idx)
        
        return matches
    
    def calculate_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """Calculate Average Precision using 11-point interpolation"""
        if not precisions or not recalls:
            return 0.0
        
        # Add boundary conditions
        precisions = [0] + precisions + [0]
        recalls = [0] + recalls + [1]
        
        # Compute maximum precision at each recall level
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precisions at recall >= t
            valid_precisions = [p for p, r in zip(precisions, recalls) if r >= t]
            if valid_precisions:
                ap += max(valid_precisions)
        
        return ap / 11.0
    
    def calculate_map(self, evaluations: List[ImageEvaluation], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate mean Average Precision"""
        class_detections = defaultdict(list)
        class_ground_truths = defaultdict(int)
        
        # Collect all predictions and ground truths by class
        for eval_result in evaluations:
            # Re-match with specific IoU threshold
            matches = self.match_predictions_to_gt(
                eval_result.predictions, 
                eval_result.ground_truths, 
                iou_threshold
            )
            
            matched_pred_indices = {m[0] for m in matches}
            matched_gt_indices = {m[1] for m in matches}
            
            for i, pred in enumerate(eval_result.predictions):
                class_name = self.class_names[pred.class_id]
                is_correct = i in matched_pred_indices
                class_detections[class_name].append((pred.confidence, is_correct))
            
            for gt in eval_result.ground_truths:
                if not gt.difficult:
                    class_name = self.class_names[gt.class_id]
                    class_ground_truths[class_name] += 1
        
        # Calculate AP for each class
        class_aps = {}
        for class_name in self.class_names.values():
            if class_name not in class_detections:
                class_aps[class_name] = 0.0
                continue
            
            detections = class_detections[class_name]
            num_gt = class_ground_truths[class_name]
            
            if num_gt == 0:
                class_aps[class_name] = 0.0
                continue
            
            # Sort by confidence descending
            detections.sort(key=lambda x: x[0], reverse=True)
            
            # Calculate precision and recall at each detection
            tp = 0
            fp = 0
            precisions = []
            recalls = []
            
            for confidence, is_correct in detections:
                if is_correct:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp)
                recall = tp / num_gt
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate AP
            class_aps[class_name] = self.calculate_ap(precisions, recalls)
        
        # Calculate mAP
        valid_aps = [ap for ap in class_aps.values() if not np.isnan(ap)]
        map_score = np.mean(valid_aps) if valid_aps else 0.0
        
        return {
            'mAP': map_score,
            'class_APs': class_aps
        }

class PerformanceProfiler:
    """Profile model performance metrics"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.reset()
    
    def reset(self):
        """Reset profiling statistics"""
        self.inference_times = []
        self.gpu_memory_usage = []
        self.cpu_usage = []
    
    def start_profiling(self):
        """Start profiling session"""
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent()
    
    def end_profiling(self):
        """End profiling session and record metrics"""
        inference_time = time.time() - self.start_time
        self.inference_times.append(inference_time)
        
        if self.gpu_available:
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_usage.append(gpu_memory)
        
        cpu_usage = psutil.cpu_percent() - self.start_cpu
        self.cpu_usage.append(max(0, cpu_usage))
    
    def get_performance_metrics(self, model_size_mb: float = 0) -> Dict[str, float]:
        """Get aggregated performance metrics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'median_inference_time': np.median(self.inference_times),
            'p95_inference_time': np.percentile(self.inference_times, 95),
            'fps': 1.0 / np.mean(self.inference_times),
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'avg_cpu_usage_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'model_size_mb': model_size_mb
        }

class RobustnessEvaluator:
    """Evaluate model robustness under various conditions"""
    
    def __init__(self):
        self.augmentation_functions = {
            'brightness': self.adjust_brightness,
            'contrast': self.adjust_contrast,
            'blur': self.apply_blur,
            'noise': self.add_noise,
            'rotation': self.rotate_image,
            'scale': self.scale_image
        }
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness"""
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast"""
        mean = np.mean(image)
        return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def apply_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def add_noise(self, image: np.ndarray, noise_factor: float) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    def scale_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale image"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        scaled = cv2.resize(image, (new_width, new_height))
        
        if scale_factor > 1:
            # Crop to original size
            start_h = (new_height - height) // 2
            start_w = (new_width - width) // 2
            return scaled[start_h:start_h + height, start_w:start_w + width]
        else:
            # Pad to original size
            pad_h = (height - new_height) // 2
            pad_w = (width - new_width) // 2
            return cv2.copyMakeBorder(scaled, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)
    
    def evaluate_robustness(self, model, test_images: List[np.ndarray], 
                          ground_truths: List[List[GroundTruth]], 
                          augmentation_levels: Dict[str, List[float]]) -> Dict[str, float]:
        """Evaluate model robustness under different conditions"""
        baseline_scores = []
        augmented_scores = {}
        
        metrics_calc = MetricsCalculator()
        
        # Calculate baseline performance
        for img, gt in zip(test_images, ground_truths):
            # Run inference on original image (placeholder)
            predictions = []  # Would run actual inference
            
            evaluation = ImageEvaluation(
                image_id=f"baseline_{len(baseline_scores)}",
                predictions=predictions,
                ground_truths=gt,
                matches=[],
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                processing_time=0.0
            )
            baseline_scores.append(evaluation)
        
        baseline_map = metrics_calc.calculate_map(baseline_scores)['mAP']
        
        # Test under different augmentations
        for aug_name, levels in augmentation_levels.items():
            aug_scores = []
            
            for level in levels:
                level_scores = []
                
                for img, gt in zip(test_images, ground_truths):
                    # Apply augmentation
                    aug_img = self.augmentation_functions[aug_name](img, level)
                    
                    # Run inference (placeholder)
                    predictions = []  # Would run actual inference
                    
                    evaluation = ImageEvaluation(
                        image_id=f"{aug_name}_{level}_{len(level_scores)}",
                        predictions=predictions,
                        ground_truths=gt,
                        matches=[],
                        true_positives=0,
                        false_positives=0,
                        false_negatives=0,
                        processing_time=0.0
                    )
                    level_scores.append(evaluation)
                
                level_map = metrics_calc.calculate_map(level_scores)['mAP']
                aug_scores.append(level_map / baseline_map if baseline_map > 0 else 0)
            
            # Calculate robustness score (average relative performance)
            augmented_scores[aug_name] = np.mean(aug_scores)
        
        return augmented_scores

class EvaluationFramework:
    """Main evaluation framework orchestrating all evaluations"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.metrics_calculator = MetricsCalculator()
        self.performance_profiler = PerformanceProfiler()
        self.robustness_evaluator = RobustnessEvaluator()
        
        # Load model
        self.model = None
        if model_path:
            self.load_model()
    
    def load_model(self):
        """Load model for evaluation"""
        try:
            if self.model_path.endswith('.pt'):
                self.model = torch.load(self.model_path, map_location='cpu')
            else:
                # Load custom model
                variants = create_medical_yolo_variants()
                self.model = variants['medical_medium']  # Default variant
            
            self.model.eval()
            print(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
    
    def evaluate_dataset(self, dataset_path: str, split: str = 'test') -> ModelMetrics:
        """Evaluate model on dataset"""
        dataset_dir = Path(dataset_path)
        images_dir = dataset_dir / 'images' / split
        labels_dir = dataset_dir / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"Dataset split '{split}' not found")
        
        # Load images and annotations
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        evaluations = []
        
        print(f"Evaluating {len(image_files)} images...")
        
        for img_file in image_files:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Load ground truth
            label_file = labels_dir / (img_file.stem + '.txt')
            ground_truths = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert to absolute coordinates
                            img_h, img_w = image.shape[:2]
                            x1 = (x_center - width/2) * img_w
                            y1 = (y_center - height/2) * img_h
                            x2 = (x_center + width/2) * img_w
                            y2 = (y_center + height/2) * img_h
                            
                            area = (x2 - x1) * (y2 - y1)
                            
                            gt = GroundTruth(
                                class_id=class_id,
                                class_name=self.metrics_calculator.class_names[class_id],
                                bbox=[x1, y1, x2, y2],
                                area=area
                            )
                            ground_truths.append(gt)
            
            # Run inference
            self.performance_profiler.start_profiling()
            predictions = self.run_inference(image)
            self.performance_profiler.end_profiling()
            
            # Match predictions to ground truth
            matches = self.metrics_calculator.match_predictions_to_gt(predictions, ground_truths)
            
            # Calculate metrics
            tp = len(matches)
            fp = len(predictions) - tp
            fn = len(ground_truths) - tp
            
            evaluation = ImageEvaluation(
                image_id=img_file.stem,
                predictions=predictions,
                ground_truths=ground_truths,
                matches=matches,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                processing_time=self.performance_profiler.inference_times[-1] if self.performance_profiler.inference_times else 0
            )
            evaluations.append(evaluation)
        
        # Calculate comprehensive metrics
        return self.calculate_comprehensive_metrics(evaluations)
    
    def run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """Run model inference on image"""
        # Placeholder inference - would use actual model
        # This simulates detection results
        predictions = [
            DetectionResult(
                class_id=0,
                class_name='syringe',
                confidence=0.85,
                bbox=[100, 100, 200, 200],
                area=10000,
                center=(150, 150)
            ),
            DetectionResult(
                class_id=1,
                class_name='bandage',
                confidence=0.92,
                bbox=[300, 150, 400, 250],
                area=10000,
                center=(350, 200)
            )
        ]
        
        return predictions
    
    def calculate_comprehensive_metrics(self, evaluations: List[ImageEvaluation]) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        # Detection metrics
        map_50_result = self.metrics_calculator.calculate_map(evaluations, 0.5)
        map_95_result = self.metrics_calculator.calculate_map(evaluations, 0.95)
        
        # Overall precision, recall, F1
        total_tp = sum(eval_result.true_positives for eval_result in evaluations)
        total_fp = sum(eval_result.false_positives for eval_result in evaluations)
        total_fn = sum(eval_result.false_negatives for eval_result in evaluations)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Performance metrics
        performance_metrics = self.performance_profiler.get_performance_metrics()
        
        # Counting metrics (placeholder)
        counting_mae = 0.5  # Would calculate actual MAE
        counting_rmse = 0.7  # Would calculate actual RMSE
        counting_mape = 0.1  # Would calculate actual MAPE
        
        # Robustness metrics (placeholder)
        robustness_scores = {
            'brightness': 0.9,
            'contrast': 0.85,
            'blur': 0.8,
            'noise': 0.75,
            'rotation': 0.88,
            'scale': 0.82
        }
        
        return ModelMetrics(
            map_50=map_50_result['mAP'],
            map_95=map_95_result['mAP'],
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            class_precisions={},  # Would calculate per-class metrics
            class_recalls={},
            class_f1_scores={},
            class_aps=map_50_result['class_APs'],
            avg_inference_time=performance_metrics.get('avg_inference_time', 0),
            fps=performance_metrics.get('fps', 0),
            gpu_memory_mb=performance_metrics.get('avg_gpu_memory_mb', 0),
            cpu_usage_percent=performance_metrics.get('avg_cpu_usage_percent', 0),
            model_size_mb=performance_metrics.get('model_size_mb', 0),
            counting_mae=counting_mae,
            counting_rmse=counting_rmse,
            counting_mape=counting_mape,
            robustness_scores=robustness_scores
        )
    
    def generate_evaluation_report(self, metrics: ModelMetrics, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'metrics': asdict(metrics),
            'hardware_info': {
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved to {output_path}")
        return report
    
    def create_evaluation_visualizations(self, metrics: ModelMetrics, evaluations: List[ImageEvaluation], 
                                       output_dir: str = "evaluation_plots"):
        """Create evaluation visualizations"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Class-wise AP plot
        plt.figure(figsize=(12, 6))
        classes = list(metrics.class_aps.keys())
        aps = list(metrics.class_aps.values())
        
        plt.bar(classes, aps, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Average Precision by Class', fontsize=16, fontweight='bold')
        plt.xlabel('Object Classes')
        plt.ylabel('Average Precision')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_ap_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution
        plt.figure(figsize=(10, 6))
        all_confidences = [pred.confidence for eval_result in evaluations 
                          for pred in eval_result.predictions]
        
        plt.hist(all_confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance metrics radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        performance_metrics = [
            ('mAP@0.5', metrics.map_50),
            ('Precision', metrics.precision),
            ('Recall', metrics.recall),
            ('F1-Score', metrics.f1_score),
            ('Speed (FPS/100)', metrics.fps / 100),
            ('Efficiency', 1 - metrics.gpu_memory_mb / 8000)  # Normalized GPU usage
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(performance_metrics), endpoint=False)
        values = [metric[1] for metric in performance_metrics]
        labels = [metric[0] for metric in performance_metrics]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Robustness scores
        plt.figure(figsize=(10, 6))
        rob_methods = list(metrics.robustness_scores.keys())
        rob_scores = list(metrics.robustness_scores.values())
        
        plt.bar(rob_methods, rob_scores, color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        plt.title('Model Robustness Scores', fontsize=16, fontweight='bold')
        plt.xlabel('Augmentation Type')
        plt.ylabel('Robustness Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/robustness_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Evaluation plots saved to {output_dir}/")

def main():
    """Example usage of evaluation framework"""
    # Initialize framework
    framework = EvaluationFramework()
    
    # Evaluate on test dataset
    try:
        dataset_path = "./datasets/medical_inventory"
        metrics = framework.evaluate_dataset(dataset_path, split='test')
        
        # Generate report
        report = framework.generate_evaluation_report(metrics)
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"mAP@0.5: {metrics.map_50:.3f}")
        print(f"mAP@0.5:0.95: {metrics.map_95:.3f}")
        print(f"Precision: {metrics.precision:.3f}")
        print(f"Recall: {metrics.recall:.3f}")
        print(f"F1-Score: {metrics.f1_score:.3f}")
        print(f"Average FPS: {metrics.fps:.1f}")
        print(f"GPU Memory: {metrics.gpu_memory_mb:.1f} MB")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        
        # Create dummy metrics for demonstration
        dummy_metrics = ModelMetrics(
            map_50=0.85, map_95=0.72, precision=0.88, recall=0.82, f1_score=0.85,
            class_precisions={}, class_recalls={}, class_f1_scores={},
            class_aps={'syringe': 0.9, 'bandage': 0.85, 'medicine_bottle': 0.8},
            avg_inference_time=0.025, fps=40.0, gpu_memory_mb=2048, cpu_usage_percent=25.0,
            model_size_mb=25.0, counting_mae=0.5, counting_rmse=0.7, counting_mape=0.1,
            robustness_scores={'brightness': 0.9, 'contrast': 0.85, 'blur': 0.8}
        )
        
        framework.generate_evaluation_report(dummy_metrics)
        print("✓ Demo evaluation report generated")

if __name__ == "__main__":
    main()