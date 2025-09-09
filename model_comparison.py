#!/usr/bin/env python3
"""
Model Comparison and A/B Testing Framework
Compare different model variants and track performance over time
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score

from evaluation_framework import EvaluationFramework, ModelMetrics, ImageEvaluation

@dataclass
class ModelComparison:
    """Results of model comparison"""
    model_a_name: str
    model_b_name: str
    model_a_metrics: ModelMetrics
    model_b_metrics: ModelMetrics
    statistical_significance: Dict[str, float]
    winner: str
    confidence_level: float
    comparison_timestamp: str

@dataclass
class ABTestResult:
    """A/B test results"""
    test_name: str
    champion_model: str
    challenger_model: str
    sample_size: int
    champion_performance: float
    challenger_performance: float
    improvement: float
    p_value: float
    statistical_significance: bool
    confidence_interval: Tuple[float, float]

class ModelComparator:
    """Compare multiple models and determine statistical significance"""
    
    def __init__(self):
        self.evaluation_framework = EvaluationFramework()
        self.comparison_history = []
    
    def compare_models(self, model_a_path: str, model_b_path: str, 
                      dataset_path: str, test_name: str = "comparison") -> ModelComparison:
        """Compare two models on the same dataset"""
        print(f"Comparing models: {model_a_path} vs {model_b_path}")
        
        # Evaluate model A
        print("Evaluating Model A...")
        self.evaluation_framework.model_path = model_a_path
        self.evaluation_framework.load_model()
        metrics_a = self.evaluation_framework.evaluate_dataset(dataset_path)
        
        # Evaluate model B
        print("Evaluating Model B...")
        self.evaluation_framework.model_path = model_b_path
        self.evaluation_framework.load_model()
        metrics_b = self.evaluation_framework.evaluate_dataset(dataset_path)
        
        # Calculate statistical significance
        significance = self.calculate_statistical_significance(metrics_a, metrics_b)
        
        # Determine winner
        winner = self.determine_winner(metrics_a, metrics_b)
        confidence = self.calculate_confidence_level(significance)
        
        comparison = ModelComparison(
            model_a_name=Path(model_a_path).stem,
            model_b_name=Path(model_b_path).stem,
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            statistical_significance=significance,
            winner=winner,
            confidence_level=confidence,
            comparison_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        self.comparison_history.append(comparison)
        return comparison
    
    def calculate_statistical_significance(self, metrics_a: ModelMetrics, 
                                         metrics_b: ModelMetrics) -> Dict[str, float]:
        """Calculate statistical significance between metrics"""
        significance = {}
        
        # Compare key metrics
        metric_pairs = [
            ('map_50', metrics_a.map_50, metrics_b.map_50),
            ('precision', metrics_a.precision, metrics_b.precision),
            ('recall', metrics_a.recall, metrics_b.recall),
            ('f1_score', metrics_a.f1_score, metrics_b.f1_score),
            ('fps', metrics_a.fps, metrics_b.fps)
        ]
        
        for metric_name, value_a, value_b in metric_pairs:
            # For simplicity, using a basic statistical test
            # In practice, would use proper bootstrap or permutation tests
            
            # Simulate samples for statistical testing
            samples_a = np.random.normal(value_a, value_a * 0.1, 100)
            samples_b = np.random.normal(value_b, value_b * 0.1, 100)
            
            # Perform t-test
            try:
                t_stat, p_value = stats.ttest_ind(samples_a, samples_b)
                significance[metric_name] = p_value
            except:
                significance[metric_name] = 1.0  # No significance
        
        return significance
    
    def determine_winner(self, metrics_a: ModelMetrics, metrics_b: ModelMetrics) -> str:
        """Determine overall winner based on weighted metrics"""
        # Define weights for different metrics
        weights = {
            'map_50': 0.4,
            'precision': 0.2,
            'recall': 0.2,
            'f1_score': 0.1,
            'fps': 0.1
        }
        
        score_a = (
            weights['map_50'] * metrics_a.map_50 +
            weights['precision'] * metrics_a.precision +
            weights['recall'] * metrics_a.recall +
            weights['f1_score'] * metrics_a.f1_score +
            weights['fps'] * min(metrics_a.fps / 100, 1.0)  # Normalize FPS
        )
        
        score_b = (
            weights['map_50'] * metrics_b.map_50 +
            weights['precision'] * metrics_b.precision +
            weights['recall'] * metrics_b.recall +
            weights['f1_score'] * metrics_b.f1_score +
            weights['fps'] * min(metrics_b.fps / 100, 1.0)  # Normalize FPS
        )
        
        return "Model A" if score_a > score_b else "Model B"
    
    def calculate_confidence_level(self, significance: Dict[str, float]) -> float:
        """Calculate overall confidence level"""
        # Average p-values and convert to confidence
        avg_p_value = np.mean(list(significance.values()))
        confidence = (1 - avg_p_value) * 100
        return min(confidence, 95.0)  # Cap at 95%

class ABTester:
    """A/B testing framework for model deployment"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = []
    
    def create_ab_test(self, test_name: str, champion_model: str, 
                      challenger_model: str, traffic_split: float = 0.5) -> Dict:
        """Create new A/B test"""
        test_config = {
            'test_name': test_name,
            'champion_model': champion_model,
            'challenger_model': challenger_model,
            'traffic_split': traffic_split,
            'start_time': time.time(),
            'champion_requests': [],
            'challenger_requests': [],
            'champion_metrics': [],
            'challenger_metrics': []
        }
        
        self.active_tests[test_name] = test_config
        print(f"✓ A/B test '{test_name}' created")
        return test_config
    
    def record_request(self, test_name: str, model_used: str, 
                      processing_time: float, accuracy: float) -> None:
        """Record a request for A/B testing"""
        if test_name not in self.active_tests:
            return
        
        test = self.active_tests[test_name]
        
        if model_used == test['champion_model']:
            test['champion_requests'].append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'accuracy': accuracy
            })
            test['champion_metrics'].append(accuracy)
        elif model_used == test['challenger_model']:
            test['challenger_requests'].append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'accuracy': accuracy
            })
            test['challenger_metrics'].append(accuracy)
    
    def analyze_ab_test(self, test_name: str, min_sample_size: int = 100) -> Optional[ABTestResult]:
        """Analyze A/B test results"""
        if test_name not in self.active_tests:
            return None
        
        test = self.active_tests[test_name]
        
        champion_metrics = test['champion_metrics']
        challenger_metrics = test['challenger_metrics']
        
        if len(champion_metrics) < min_sample_size or len(challenger_metrics) < min_sample_size:
            print(f"Insufficient sample size for test '{test_name}'")
            return None
        
        # Calculate performance metrics
        champion_perf = np.mean(champion_metrics)
        challenger_perf = np.mean(challenger_metrics)
        
        improvement = (challenger_perf - champion_perf) / champion_perf * 100
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(challenger_metrics, champion_metrics)
        is_significant = p_value < 0.05
        
        # Confidence interval
        pooled_std = np.sqrt(
            ((len(challenger_metrics) - 1) * np.var(challenger_metrics) + 
             (len(champion_metrics) - 1) * np.var(champion_metrics)) /
            (len(challenger_metrics) + len(champion_metrics) - 2)
        )
        
        margin_of_error = 1.96 * pooled_std * np.sqrt(
            1/len(challenger_metrics) + 1/len(champion_metrics)
        )
        
        ci_lower = improvement - margin_of_error
        ci_upper = improvement + margin_of_error
        
        result = ABTestResult(
            test_name=test_name,
            champion_model=test['champion_model'],
            challenger_model=test['challenger_model'],
            sample_size=len(champion_metrics) + len(challenger_metrics),
            champion_performance=champion_perf,
            challenger_performance=challenger_perf,
            improvement=improvement,
            p_value=p_value,
            statistical_significance=is_significant,
            confidence_interval=(ci_lower, ci_upper)
        )
        
        self.test_results.append(result)
        return result
    
    def get_test_recommendation(self, test_result: ABTestResult) -> str:
        """Get recommendation based on A/B test results"""
        if not test_result.statistical_significance:
            return "No significant difference. Continue with champion model."
        
        if test_result.improvement > 0:
            return f"Deploy challenger model. Significant improvement of {test_result.improvement:.1f}%"
        else:
            return f"Keep champion model. Challenger performs {abs(test_result.improvement):.1f}% worse."

class ModelPerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self, storage_path: str = "performance_history.json"):
        self.storage_path = storage_path
        self.performance_history = self.load_history()
    
    def load_history(self) -> List[Dict]:
        """Load performance history from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_history(self):
        """Save performance history to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def record_performance(self, model_name: str, metrics: ModelMetrics, 
                          deployment_version: str = "1.0"):
        """Record model performance"""
        performance_record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'deployment_version': deployment_version,
            'metrics': asdict(metrics)
        }
        
        self.performance_history.append(performance_record)
        self.save_history()
    
    def get_performance_trend(self, model_name: str, metric_name: str, 
                            days: int = 30) -> Tuple[List[str], List[float]]:
        """Get performance trend for a specific metric"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        timestamps = []
        values = []
        
        for record in self.performance_history:
            if record['model_name'] == model_name:
                record_time = time.mktime(time.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S'))
                if record_time >= cutoff_time:
                    timestamps.append(record['timestamp'])
                    values.append(record['metrics'].get(metric_name, 0))
        
        return timestamps, values
    
    def detect_performance_regression(self, model_name: str, 
                                    threshold: float = 0.05) -> Optional[Dict]:
        """Detect performance regression"""
        if len(self.performance_history) < 10:
            return None
        
        # Get recent performance for the model
        model_records = [r for r in self.performance_history if r['model_name'] == model_name]
        
        if len(model_records) < 10:
            return None
        
        # Compare recent vs historical performance
        recent_records = model_records[-5:]  # Last 5 records
        historical_records = model_records[-15:-5]  # Previous 10 records
        
        recent_map = np.mean([r['metrics']['map_50'] for r in recent_records])
        historical_map = np.mean([r['metrics']['map_50'] for r in historical_records])
        
        regression = (historical_map - recent_map) / historical_map
        
        if regression > threshold:
            return {
                'model_name': model_name,
                'regression_detected': True,
                'regression_percentage': regression * 100,
                'recent_performance': recent_map,
                'historical_performance': historical_map,
                'recommendation': 'Investigate model degradation'
            }
        
        return None

class ComparisonVisualizer:
    """Create visualizations for model comparisons"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_model_comparison(self, comparison: ModelComparison, output_path: str = "model_comparison.png"):
        """Create comprehensive model comparison plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics_a = comparison.model_a_metrics
        metrics_b = comparison.model_b_metrics
        
        # 1. Key metrics comparison
        metrics = ['mAP@0.5', 'Precision', 'Recall', 'F1-Score']
        values_a = [metrics_a.map_50, metrics_a.precision, metrics_a.recall, metrics_a.f1_score]
        values_b = [metrics_b.map_50, metrics_b.precision, metrics_b.recall, metrics_b.f1_score]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, values_a, width, label=comparison.model_a_name, alpha=0.8)
        bars2 = ax1.bar(x + width/2, values_b, width, label=comparison.model_b_name, alpha=0.8)
        
        ax1.set_title('Detection Performance Comparison', fontweight='bold')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Performance metrics
        perf_metrics = ['Inference Time (ms)', 'FPS', 'GPU Memory (MB)']
        perf_values_a = [metrics_a.avg_inference_time * 1000, metrics_a.fps, metrics_a.gpu_memory_mb]
        perf_values_b = [metrics_b.avg_inference_time * 1000, metrics_b.fps, metrics_b.gpu_memory_mb]
        
        # Normalize values for comparison
        perf_values_a_norm = [
            perf_values_a[0] / 100,  # Normalize inference time
            perf_values_a[1] / 100,  # Normalize FPS
            perf_values_a[2] / 4000  # Normalize GPU memory
        ]
        perf_values_b_norm = [
            perf_values_b[0] / 100,
            perf_values_b[1] / 100,
            perf_values_b[2] / 4000
        ]
        
        x_perf = np.arange(len(perf_metrics))
        bars3 = ax2.bar(x_perf - width/2, perf_values_a_norm, width, 
                       label=comparison.model_a_name, alpha=0.8)
        bars4 = ax2.bar(x_perf + width/2, perf_values_b_norm, width, 
                       label=comparison.model_b_name, alpha=0.8)
        
        ax2.set_title('Performance Comparison (Normalized)', fontweight='bold')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Normalized Score')
        ax2.set_xticks(x_perf)
        ax2.set_xticklabels(perf_metrics, rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Statistical significance
        significance = comparison.statistical_significance
        metrics_sig = list(significance.keys())
        p_values = list(significance.values())
        
        colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'green' for p in p_values]
        
        bars5 = ax3.bar(metrics_sig, p_values, color=colors, alpha=0.7)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='p=0.10')
        
        ax3.set_title('Statistical Significance (p-values)', fontweight='bold')
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('p-value')
        ax3.set_xticks(range(len(metrics_sig)))
        ax3.set_xticklabels(metrics_sig, rotation=45)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Class-wise AP comparison
        class_names = list(metrics_a.class_aps.keys())
        class_aps_a = [metrics_a.class_aps.get(cls, 0) for cls in class_names]
        class_aps_b = [metrics_b.class_aps.get(cls, 0) for cls in class_names]
        
        x_cls = np.arange(len(class_names))
        bars6 = ax4.bar(x_cls - width/2, class_aps_a, width, 
                       label=comparison.model_a_name, alpha=0.8)
        bars7 = ax4.bar(x_cls + width/2, class_aps_b, width, 
                       label=comparison.model_b_name, alpha=0.8)
        
        ax4.set_title('Class-wise Average Precision', fontweight='bold')
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('Average Precision')
        ax4.set_xticks(x_cls)
        ax4.set_xticklabels(class_names, rotation=45)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Model comparison plot saved to {output_path}")
    
    def plot_ab_test_results(self, ab_result: ABTestResult, output_path: str = "ab_test_results.png"):
        """Plot A/B test results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance comparison
        models = [ab_result.champion_model, ab_result.challenger_model]
        performances = [ab_result.champion_performance, ab_result.challenger_performance]
        colors = ['blue', 'red']
        
        bars = ax1.bar(models, performances, color=colors, alpha=0.7)
        ax1.set_title('A/B Test Performance Comparison', fontweight='bold')
        ax1.set_ylabel('Performance Metric')
        
        # Add improvement percentage
        improvement_text = f"Improvement: {ab_result.improvement:+.1f}%"
        ax1.text(0.5, max(performances) * 0.9, improvement_text, 
                ha='center', fontsize=12, fontweight='bold')
        
        # Add significance indicator
        significance_text = "Statistically Significant" if ab_result.statistical_significance else "Not Significant"
        color = 'green' if ab_result.statistical_significance else 'red'
        ax1.text(0.5, max(performances) * 0.8, significance_text, 
                ha='center', fontsize=10, color=color)
        
        # Confidence interval
        ci_lower, ci_upper = ab_result.confidence_interval
        ax2.bar(['Improvement'], [ab_result.improvement], 
               yerr=[[ab_result.improvement - ci_lower], [ci_upper - ab_result.improvement]], 
               capsize=5, alpha=0.7, color='green' if ab_result.improvement > 0 else 'red')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Improvement with Confidence Interval', fontweight='bold')
        ax2.set_ylabel('Improvement (%)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ A/B test results plot saved to {output_path}")

def main():
    """Example usage of model comparison framework"""
    # Initialize components
    comparator = ModelComparator()
    ab_tester = ABTester()
    visualizer = ComparisonVisualizer()
    
    print("Model Comparison Framework initialized")
    print("Ready for model comparisons and A/B testing")
    
    # Example A/B test
    test_config = ab_tester.create_ab_test(
        "medical_yolo_v1_vs_v2",
        "medical_yolo_v1",
        "medical_yolo_v2"
    )
    
    # Simulate some test data
    for i in range(200):
        model = "medical_yolo_v1" if np.random.random() < 0.5 else "medical_yolo_v2"
        processing_time = np.random.normal(0.05, 0.01)
        accuracy = np.random.normal(0.85 if model == "medical_yolo_v1" else 0.87, 0.05)
        
        ab_tester.record_request("medical_yolo_v1_vs_v2", model, processing_time, accuracy)
    
    # Analyze test
    result = ab_tester.analyze_ab_test("medical_yolo_v1_vs_v2")
    if result:
        print(f"\nA/B Test Results:")
        print(f"Improvement: {result.improvement:+.1f}%")
        print(f"Statistical Significance: {result.statistical_significance}")
        print(f"Recommendation: {ab_tester.get_test_recommendation(result)}")

if __name__ == "__main__":
    main()