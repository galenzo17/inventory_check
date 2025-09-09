#!/usr/bin/env python3
"""
Tests for evaluation framework
"""

import pytest
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from dataclasses import asdict

from evaluation_framework import (
    EvaluationFramework,
    MetricsCalculator, 
    PerformanceProfiler,
    RobustnessEvaluator,
    DetectionResult,
    GroundTruth,
    ImageEvaluation,
    ModelMetrics
)
from model_comparison import ModelComparator, ABTester, ModelPerformanceTracker

class TestDetectionResult:
    """Test DetectionResult dataclass"""
    
    def test_detection_result_creation(self):
        """Test creating DetectionResult"""
        result = DetectionResult(
            class_id=0,
            class_name="syringe",
            confidence=0.85,
            bbox=[100, 150, 200, 250],
            area=10000,
            center=(150, 200)
        )
        
        assert result.class_id == 0
        assert result.class_name == "syringe"
        assert result.confidence == 0.85
        assert result.bbox == [100, 150, 200, 250]
        assert result.area == 10000
        assert result.center == (150, 200)

class TestGroundTruth:
    """Test GroundTruth dataclass"""
    
    def test_ground_truth_creation(self):
        """Test creating GroundTruth"""
        gt = GroundTruth(
            class_id=1,
            class_name="bandage",
            bbox=[50, 75, 150, 175],
            area=10000,
            difficult=False
        )
        
        assert gt.class_id == 1
        assert gt.class_name == "bandage" 
        assert gt.bbox == [50, 75, 150, 175]
        assert gt.area == 10000
        assert gt.difficult is False

class TestMetricsCalculator:
    """Test MetricsCalculator functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.calculator = MetricsCalculator()
    
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        assert len(self.calculator.iou_thresholds) == 10
        assert 0.5 in self.calculator.iou_thresholds
        assert 0.95 in self.calculator.iou_thresholds
        assert len(self.calculator.class_names) == 10
        assert self.calculator.class_names[0] == 'syringe'
    
    def test_calculate_iou_perfect_match(self):
        """Test IoU calculation for perfect match"""
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        
        iou = self.calculator.calculate_iou(box1, box2)
        
        assert abs(iou - 1.0) < 1e-6
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for no overlap"""
        box1 = [100, 100, 200, 200]
        box2 = [300, 300, 400, 400]
        
        iou = self.calculator.calculate_iou(box1, box2)
        
        assert abs(iou - 0.0) < 1e-6
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation for partial overlap"""
        box1 = [100, 100, 200, 200]  # 100x100 box
        box2 = [150, 150, 250, 250]  # 100x100 box, 50x50 overlap
        
        iou = self.calculator.calculate_iou(box1, box2)
        
        # Expected: intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 1e-6
    
    def test_match_predictions_to_gt_empty(self):
        """Test matching with empty inputs"""
        matches = self.calculator.match_predictions_to_gt([], [], 0.5)
        assert matches == []
        
        predictions = [DetectionResult(0, "syringe", 0.8, [100, 100, 200, 200], 10000, (150, 150))]
        matches = self.calculator.match_predictions_to_gt(predictions, [], 0.5)
        assert matches == []
        
        ground_truths = [GroundTruth(0, "syringe", [100, 100, 200, 200], 10000)]
        matches = self.calculator.match_predictions_to_gt([], ground_truths, 0.5)
        assert matches == []
    
    def test_match_predictions_to_gt_perfect_match(self):
        """Test matching with perfect prediction-GT match"""
        pred = DetectionResult(0, "syringe", 0.8, [100, 100, 200, 200], 10000, (150, 150))
        gt = GroundTruth(0, "syringe", [100, 100, 200, 200], 10000)
        
        matches = self.calculator.match_predictions_to_gt([pred], [gt], 0.5)
        
        assert len(matches) == 1
        assert matches[0] == (0, 0, 1.0)  # pred_idx, gt_idx, iou
    
    def test_match_predictions_to_gt_class_mismatch(self):
        """Test matching with class mismatch"""
        pred = DetectionResult(0, "syringe", 0.8, [100, 100, 200, 200], 10000, (150, 150))
        gt = GroundTruth(1, "bandage", [100, 100, 200, 200], 10000)  # Different class
        
        matches = self.calculator.match_predictions_to_gt([pred], [gt], 0.5)
        
        assert matches == []  # No match due to class mismatch
    
    def test_calculate_ap_empty(self):
        """Test AP calculation with empty inputs"""
        ap = self.calculator.calculate_ap([], [])
        assert ap == 0.0
    
    def test_calculate_ap_perfect_precision(self):
        """Test AP calculation with perfect precision"""
        precisions = [1.0, 1.0, 1.0, 1.0]
        recalls = [0.25, 0.5, 0.75, 1.0]
        
        ap = self.calculator.calculate_ap(precisions, recalls)
        
        assert ap == 1.0
    
    def test_calculate_map_empty(self):
        """Test mAP calculation with empty evaluations"""
        result = self.calculator.calculate_map([], 0.5)
        
        assert result['mAP'] == 0.0
        assert isinstance(result['class_APs'], dict)

class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = PerformanceProfiler()
    
    def test_profiler_initialization(self):
        """Test profiler initialization"""
        assert hasattr(self.profiler, 'gpu_available')
        assert isinstance(self.profiler.inference_times, list)
        assert len(self.profiler.inference_times) == 0
    
    def test_reset_profiling(self):
        """Test profiling reset"""
        self.profiler.inference_times = [0.1, 0.2]
        self.profiler.reset()
        
        assert len(self.profiler.inference_times) == 0
    
    def test_profiling_session(self):
        """Test complete profiling session"""
        import time
        
        self.profiler.start_profiling()
        time.sleep(0.01)  # Simulate some work
        self.profiler.end_profiling()
        
        assert len(self.profiler.inference_times) == 1
        assert self.profiler.inference_times[0] > 0
    
    def test_get_performance_metrics_empty(self):
        """Test getting metrics with no data"""
        metrics = self.profiler.get_performance_metrics()
        
        assert metrics == {}
    
    def test_get_performance_metrics_with_data(self):
        """Test getting metrics with profiled data"""
        # Add some mock data
        self.profiler.inference_times = [0.1, 0.15, 0.12, 0.18, 0.11]
        self.profiler.gpu_memory_usage = [1000, 1200, 1100]
        self.profiler.cpu_usage = [25, 30, 28]
        
        metrics = self.profiler.get_performance_metrics(model_size_mb=50.0)
        
        assert 'avg_inference_time' in metrics
        assert 'median_inference_time' in metrics
        assert 'fps' in metrics
        assert metrics['model_size_mb'] == 50.0
        assert metrics['avg_inference_time'] == 0.132  # Mean of inference times
        assert abs(metrics['fps'] - 1/0.132) < 0.01

class TestRobustnessEvaluator:
    """Test RobustnessEvaluator functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.evaluator = RobustnessEvaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        assert len(self.evaluator.augmentation_functions) == 6
        assert 'brightness' in self.evaluator.augmentation_functions
        assert 'rotation' in self.evaluator.augmentation_functions
    
    def test_adjust_brightness(self):
        """Test brightness adjustment"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        brighter = self.evaluator.adjust_brightness(img, 1.5)
        darker = self.evaluator.adjust_brightness(img, 0.5)
        
        assert brighter.shape == img.shape
        assert darker.shape == img.shape
        assert np.mean(brighter) > np.mean(img)
        assert np.mean(darker) < np.mean(img)
    
    def test_add_noise(self):
        """Test noise addition"""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        
        noisy = self.evaluator.add_noise(img, 0.1)
        
        assert noisy.shape == img.shape
        assert not np.array_equal(noisy, img)
    
    def test_rotate_image(self):
        """Test image rotation"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        rotated = self.evaluator.rotate_image(img, 90)
        
        assert rotated.shape == img.shape
    
    def test_scale_image_up(self):
        """Test image scaling up"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        scaled = self.evaluator.scale_image(img, 1.5)
        
        assert scaled.shape == img.shape  # Should be cropped back to original size
    
    def test_scale_image_down(self):
        """Test image scaling down"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        scaled = self.evaluator.scale_image(img, 0.5)
        
        assert scaled.shape == img.shape  # Should be padded back to original size

class TestEvaluationFramework:
    """Test EvaluationFramework functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = EvaluationFramework()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        assert isinstance(self.framework.metrics_calculator, MetricsCalculator)
        assert isinstance(self.framework.performance_profiler, PerformanceProfiler)
        assert isinstance(self.framework.robustness_evaluator, RobustnessEvaluator)
    
    @patch('evaluation_framework.create_medical_yolo_variants')
    def test_load_model(self, mock_variants):
        """Test model loading"""
        mock_model = Mock()
        mock_variants.return_value = {'medical_medium': mock_model}
        
        framework = EvaluationFramework(model_path="test_model.pt")
        
        # Should attempt to load the model
        assert framework.model_path == "test_model.pt"
    
    def test_run_inference_mock(self):
        """Test inference with mock model"""
        predictions = self.framework.run_inference(np.zeros((100, 100, 3)))
        
        # Should return some mock predictions
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        assert isinstance(predictions[0], DetectionResult)
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation"""
        # Create mock evaluation data
        evaluations = [
            ImageEvaluation(
                image_id="test1",
                predictions=[
                    DetectionResult(0, "syringe", 0.9, [100, 100, 200, 200], 10000, (150, 150))
                ],
                ground_truths=[
                    GroundTruth(0, "syringe", [100, 100, 200, 200], 10000)
                ],
                matches=[(0, 0, 1.0)],
                true_positives=1,
                false_positives=0,
                false_negatives=0,
                processing_time=0.1
            )
        ]
        
        metrics = self.framework.calculate_comprehensive_metrics(evaluations)
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.precision == 1.0  # TP / (TP + FP) = 1 / (1 + 0)
        assert metrics.recall == 1.0     # TP / (TP + FN) = 1 / (1 + 0)
        assert metrics.f1_score == 1.0   # 2 * (P * R) / (P + R)
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation"""
        # Create dummy metrics
        metrics = ModelMetrics(
            map_50=0.85,
            map_95=0.72,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            class_precisions={},
            class_recalls={},
            class_f1_scores={},
            class_aps={'syringe': 0.9, 'bandage': 0.8},
            avg_inference_time=0.025,
            fps=40.0,
            gpu_memory_mb=2048,
            cpu_usage_percent=25.0,
            model_size_mb=25.0,
            counting_mae=0.5,
            counting_rmse=0.7,
            counting_mape=0.1,
            robustness_scores={'brightness': 0.9}
        )
        
        output_path = self.temp_dir + "/test_report.json"
        report = self.framework.generate_evaluation_report(metrics, output_path)
        
        assert Path(output_path).exists()
        assert 'evaluation_timestamp' in report
        assert 'metrics' in report
        assert 'hardware_info' in report
        
        # Verify file contents
        with open(output_path) as f:
            saved_report = json.load(f)
        
        assert saved_report['metrics']['map_50'] == 0.85

class TestModelComparator:
    """Test ModelComparator functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.comparator = ModelComparator()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_comparator_initialization(self):
        """Test comparator initialization"""
        assert isinstance(self.comparator.evaluation_framework, EvaluationFramework)
        assert isinstance(self.comparator.comparison_history, list)
        assert len(self.comparator.comparison_history) == 0
    
    def test_calculate_statistical_significance(self):
        """Test statistical significance calculation"""
        metrics_a = ModelMetrics(
            map_50=0.85, map_95=0.72, precision=0.88, recall=0.82, f1_score=0.85,
            class_precisions={}, class_recalls={}, class_f1_scores={}, class_aps={},
            avg_inference_time=0.025, fps=40.0, gpu_memory_mb=2048, cpu_usage_percent=25.0,
            model_size_mb=25.0, counting_mae=0.5, counting_rmse=0.7, counting_mape=0.1,
            robustness_scores={}
        )
        
        metrics_b = ModelMetrics(
            map_50=0.87, map_95=0.74, precision=0.90, recall=0.84, f1_score=0.87,
            class_precisions={}, class_recalls={}, class_f1_scores={}, class_aps={},
            avg_inference_time=0.030, fps=33.0, gpu_memory_mb=2500, cpu_usage_percent=30.0,
            model_size_mb=35.0, counting_mae=0.4, counting_rmse=0.6, counting_mape=0.08,
            robustness_scores={}
        )
        
        significance = self.comparator.calculate_statistical_significance(metrics_a, metrics_b)
        
        assert isinstance(significance, dict)
        assert 'map_50' in significance
        assert 'precision' in significance
        assert all(0 <= p <= 1 for p in significance.values())
    
    def test_determine_winner(self):
        """Test winner determination"""
        metrics_a = ModelMetrics(
            map_50=0.85, map_95=0.72, precision=0.88, recall=0.82, f1_score=0.85,
            class_precisions={}, class_recalls={}, class_f1_scores={}, class_aps={},
            avg_inference_time=0.025, fps=40.0, gpu_memory_mb=2048, cpu_usage_percent=25.0,
            model_size_mb=25.0, counting_mae=0.5, counting_rmse=0.7, counting_mape=0.1,
            robustness_scores={}
        )
        
        metrics_b = ModelMetrics(
            map_50=0.90, map_95=0.78, precision=0.92, recall=0.86, f1_score=0.89,  # Better
            class_precisions={}, class_recalls={}, class_f1_scores={}, class_aps={},
            avg_inference_time=0.030, fps=33.0, gpu_memory_mb=2500, cpu_usage_percent=30.0,
            model_size_mb=35.0, counting_mae=0.4, counting_rmse=0.6, counting_mape=0.08,
            robustness_scores={}
        )
        
        winner = self.comparator.determine_winner(metrics_a, metrics_b)
        
        assert winner == "Model B"  # B has better metrics

class TestABTester:
    """Test A/B testing functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.ab_tester = ABTester()
    
    def test_ab_tester_initialization(self):
        """Test A/B tester initialization"""
        assert isinstance(self.ab_tester.active_tests, dict)
        assert isinstance(self.ab_tester.test_results, list)
        assert len(self.ab_tester.active_tests) == 0
    
    def test_create_ab_test(self):
        """Test A/B test creation"""
        test_config = self.ab_tester.create_ab_test(
            "test_comparison",
            "model_v1",
            "model_v2",
            traffic_split=0.6
        )
        
        assert test_config['test_name'] == "test_comparison"
        assert test_config['champion_model'] == "model_v1"
        assert test_config['challenger_model'] == "model_v2"
        assert test_config['traffic_split'] == 0.6
        assert "test_comparison" in self.ab_tester.active_tests
    
    def test_record_request(self):
        """Test recording A/B test requests"""
        # Create test first
        self.ab_tester.create_ab_test("test", "champion", "challenger")
        
        # Record requests
        self.ab_tester.record_request("test", "champion", 0.1, 0.9)
        self.ab_tester.record_request("test", "challenger", 0.12, 0.92)
        
        test = self.ab_tester.active_tests["test"]
        assert len(test['champion_requests']) == 1
        assert len(test['challenger_requests']) == 1
        assert len(test['champion_metrics']) == 1
        assert len(test['challenger_metrics']) == 1
    
    def test_analyze_ab_test_insufficient_data(self):
        """Test A/B test analysis with insufficient data"""
        self.ab_tester.create_ab_test("test", "champion", "challenger")
        
        # Only record a few requests (less than min_sample_size)
        for i in range(5):
            self.ab_tester.record_request("test", "champion", 0.1, 0.9)
            self.ab_tester.record_request("test", "challenger", 0.12, 0.92)
        
        result = self.ab_tester.analyze_ab_test("test", min_sample_size=100)
        
        assert result is None  # Insufficient data
    
    def test_get_test_recommendation(self):
        """Test getting A/B test recommendations"""
        from model_comparison import ABTestResult
        
        # Test with significant improvement
        result_positive = ABTestResult(
            test_name="test",
            champion_model="v1",
            challenger_model="v2",
            sample_size=200,
            champion_performance=0.85,
            challenger_performance=0.90,
            improvement=5.88,
            p_value=0.01,
            statistical_significance=True,
            confidence_interval=(3.0, 8.0)
        )
        
        recommendation = self.ab_tester.get_test_recommendation(result_positive)
        assert "Deploy challenger" in recommendation
        assert "5.9%" in recommendation
        
        # Test with no significance
        result_no_sig = ABTestResult(
            test_name="test",
            champion_model="v1",
            challenger_model="v2",
            sample_size=200,
            champion_performance=0.85,
            challenger_performance=0.86,
            improvement=1.18,
            p_value=0.15,
            statistical_significance=False,
            confidence_interval=(-1.0, 3.0)
        )
        
        recommendation = self.ab_tester.get_test_recommendation(result_no_sig)
        assert "Continue with champion" in recommendation

class TestModelPerformanceTracker:
    """Test ModelPerformanceTracker functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_file = tempfile.mktemp(suffix='.json')
        self.tracker = ModelPerformanceTracker(self.temp_file)
    
    def teardown_method(self):
        """Cleanup test environment"""
        Path(self.temp_file).unlink(missing_ok=True)
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        assert self.tracker.storage_path == self.temp_file
        assert isinstance(self.tracker.performance_history, list)
    
    def test_record_performance(self):
        """Test recording performance metrics"""
        metrics = ModelMetrics(
            map_50=0.85, map_95=0.72, precision=0.88, recall=0.82, f1_score=0.85,
            class_precisions={}, class_recalls={}, class_f1_scores={}, class_aps={},
            avg_inference_time=0.025, fps=40.0, gpu_memory_mb=2048, cpu_usage_percent=25.0,
            model_size_mb=25.0, counting_mae=0.5, counting_rmse=0.7, counting_mape=0.1,
            robustness_scores={}
        )
        
        self.tracker.record_performance("test_model", metrics, "v1.0")
        
        assert len(self.tracker.performance_history) == 1
        assert Path(self.temp_file).exists()
        
        # Verify saved data
        with open(self.temp_file) as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]['model_name'] == "test_model"
        assert data[0]['deployment_version'] == "v1.0"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])