#!/usr/bin/env python3
"""
Tests for dataset management functionality
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import cv2

from dataset_manager import MedicalInventoryDataset, DataAugmentation
from annotation_tools import PreLabeler, AnnotationValidator
from data_quality_checker import DataQualityChecker

class TestMedicalInventoryDataset:
    """Test Medical Inventory Dataset management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = MedicalInventoryDataset(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_initialization(self):
        """Test dataset initialization creates proper structure"""
        assert self.dataset.dataset_root.exists()
        assert (self.dataset.dataset_root / "images" / "train").exists()
        assert (self.dataset.dataset_root / "images" / "val").exists()
        assert (self.dataset.dataset_root / "images" / "test").exists()
        assert (self.dataset.dataset_root / "labels" / "train").exists()
        assert (self.dataset.dataset_root / "labels" / "val").exists()
        assert (self.dataset.dataset_root / "labels" / "test").exists()
    
    def test_create_dataset_yaml(self):
        """Test dataset.yaml creation"""
        self.dataset.create_dataset_yaml()
        
        yaml_file = self.dataset.dataset_root / "dataset.yaml"
        assert yaml_file.exists()
        
        import yaml
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        
        assert config['nc'] == 10
        assert len(config['names']) == 10
        assert 'syringe' in config['names']
        assert config['train'] == 'images/train'
    
    def test_medical_categories(self):
        """Test medical categories are properly defined"""
        categories = self.dataset.MEDICAL_CATEGORIES
        
        assert len(categories) == 10
        assert 0 in categories
        assert categories[0] == "syringe"
        assert categories[1] == "bandage"
        assert categories[9] == "first_aid_supply"
    
    def test_validate_annotation_valid(self):
        """Test annotation validation with valid annotation"""
        # Create test image
        img_path = self.temp_dir + "/test_image.jpg"
        Image.new('RGB', (640, 480)).save(img_path)
        
        # Create valid annotation
        ann_path = self.temp_dir + "/test_annotation.txt"
        with open(ann_path, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")  # class_id x_center y_center width height
            f.write("1 0.3 0.7 0.1 0.2\n")
        
        result = self.dataset.validate_annotation(ann_path, img_path)
        assert result is True
    
    def test_validate_annotation_invalid_format(self):
        """Test annotation validation with invalid format"""
        # Create test image
        img_path = self.temp_dir + "/test_image.jpg"
        Image.new('RGB', (640, 480)).save(img_path)
        
        # Create invalid annotation (wrong number of values)
        ann_path = self.temp_dir + "/test_annotation.txt"
        with open(ann_path, 'w') as f:
            f.write("0 0.5 0.5\n")  # Missing width and height
        
        result = self.dataset.validate_annotation(ann_path, img_path)
        assert result is False
    
    def test_validate_annotation_invalid_coordinates(self):
        """Test annotation validation with invalid coordinates"""
        # Create test image
        img_path = self.temp_dir + "/test_image.jpg"
        Image.new('RGB', (640, 480)).save(img_path)
        
        # Create annotation with coordinates > 1.0
        ann_path = self.temp_dir + "/test_annotation.txt"
        with open(ann_path, 'w') as f:
            f.write("0 1.5 0.5 0.2 0.3\n")  # x_center > 1.0
        
        result = self.dataset.validate_annotation(ann_path, img_path)
        assert result is False
    
    def test_convert_to_yolo_format(self):
        """Test bounding box conversion to YOLO format"""
        bbox = {'x': 100, 'y': 150, 'width': 200, 'height': 100, 'class_id': 0}
        
        result = self.dataset.convert_to_yolo_format(bbox, 640, 480)
        
        # Expected: x_center=(100+200/2)/640=0.3125, y_center=(150+100/2)/480=0.4167
        expected_parts = result.strip().split()
        assert expected_parts[0] == "0"  # class_id
        assert abs(float(expected_parts[1]) - 0.3125) < 0.001  # x_center
        assert abs(float(expected_parts[2]) - 0.4167) < 0.001  # y_center
        assert abs(float(expected_parts[3]) - 0.3125) < 0.001  # width
        assert abs(float(expected_parts[4]) - 0.2083) < 0.001  # height
    
    def test_split_dataset(self):
        """Test dataset splitting functionality"""
        # Create some dummy images and labels
        images_dir = self.dataset.dataset_root / "images"
        labels_dir = self.dataset.dataset_root / "labels"
        
        for i in range(10):
            img_path = images_dir / f"image_{i}.jpg"
            label_path = labels_dir / f"image_{i}.txt"
            
            Image.new('RGB', (100, 100)).save(img_path)
            with open(label_path, 'w') as f:
                f.write(f"{i % 3} 0.5 0.5 0.2 0.3\n")
        
        # Split dataset
        self.dataset.split_dataset(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        
        # Check splits exist
        train_images = list((self.dataset.images_dir / "train").glob("*.jpg"))
        val_images = list((self.dataset.images_dir / "val").glob("*.jpg"))
        test_images = list((self.dataset.images_dir / "test").glob("*.jpg"))
        
        assert len(train_images) == 6  # 60% of 10
        assert len(val_images) == 2   # 20% of 10
        assert len(test_images) == 2  # 20% of 10
        
        # Check split info was saved
        split_info_file = self.dataset.splits_dir / "split_info.json"
        assert split_info_file.exists()
    
    def test_generate_statistics_empty_dataset(self):
        """Test statistics generation for empty dataset"""
        stats = self.dataset.generate_statistics()
        
        assert stats['total_images'] == 0
        assert stats['total_annotations'] == 0
        assert isinstance(stats['class_distribution'], dict)
        assert isinstance(stats['splits'], dict)
    
    def test_generate_statistics_with_data(self):
        """Test statistics generation with sample data"""
        # Create sample data
        train_dir = self.dataset.images_dir / "train"
        labels_dir = self.dataset.labels_dir / "train"
        
        # Create 3 images with annotations
        for i in range(3):
            img_path = train_dir / f"image_{i}.jpg"
            label_path = labels_dir / f"image_{i}.txt"
            
            Image.new('RGB', (100, 100)).save(img_path)
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.2 0.3\n")  # syringe
                f.write(f"1 0.3 0.7 0.1 0.2\n")  # bandage
        
        stats = self.dataset.generate_statistics()
        
        assert stats['total_images'] == 3
        assert stats['total_annotations'] == 6  # 2 annotations per image
        assert stats['class_distribution']['syringe'] == 3
        assert stats['class_distribution']['bandage'] == 3

class TestDataAugmentation:
    """Test data augmentation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.augmenter = DataAugmentation()
    
    def test_rotate_image(self):
        """Test image rotation"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        rotated = self.augmenter.rotate_image(img, 45)
        
        assert rotated.shape == img.shape
        assert not np.array_equal(rotated, img)  # Should be different
    
    def test_adjust_brightness(self):
        """Test brightness adjustment"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        brighter = self.augmenter.adjust_brightness(img, 1.5)
        darker = self.augmenter.adjust_brightness(img, 0.5)
        
        assert brighter.shape == img.shape
        assert darker.shape == img.shape
        assert np.mean(brighter) > np.mean(img)  # Should be brighter
        assert np.mean(darker) < np.mean(img)   # Should be darker
    
    def test_add_noise(self):
        """Test noise addition"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        noisy = self.augmenter.add_noise(img, 0.1)
        
        assert noisy.shape == img.shape
        assert not np.array_equal(noisy, img)  # Should be different
    
    def test_flip_horizontal(self):
        """Test horizontal flip"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        flipped = self.augmenter.flip_horizontal(img)
        
        assert flipped.shape == img.shape
        # Flipping twice should return original
        double_flipped = self.augmenter.flip_horizontal(flipped)
        assert np.array_equal(double_flipped, img)

class TestPreLabeler:
    """Test pre-labeling functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('annotation_tools.YOLO')
    def test_pre_labeler_initialization(self, mock_yolo):
        """Test pre-labeler initialization"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        pre_labeler = PreLabeler("test_model.pt")
        
        assert pre_labeler.model == mock_model
        assert pre_labeler.confidence_threshold == 0.5
        mock_yolo.assert_called_once_with("test_model.pt")
    
    @patch('annotation_tools.YOLO')
    @patch('annotation_tools.Image.open')
    def test_pre_label_image(self, mock_image_open, mock_yolo):
        """Test pre-labeling a single image"""
        # Setup mocks
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_img = Mock()
        mock_img.size = (640, 480)
        mock_image_open.return_value = mock_img
        
        # Mock YOLO results
        mock_box = Mock()
        mock_box.xyxy = [[torch.tensor([100, 150, 300, 250])]]
        mock_box.conf = [[0.85]]
        mock_box.cls = [[0]]
        
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: 'bottle'}
        
        mock_model.return_value = [mock_result]
        
        pre_labeler = PreLabeler("test_model.pt")
        
        img_path = self.temp_dir + "/test.jpg"
        out_path = self.temp_dir + "/test.txt"
        
        result = pre_labeler.pre_label_image(img_path, out_path)
        
        assert isinstance(result, dict)
        assert 'annotations_count' in result
        assert 'detections' in result
    
    def test_medical_mapping(self):
        """Test medical item mapping"""
        pre_labeler = PreLabeler()
        
        mapping = pre_labeler.medical_mapping
        
        assert 'bottle' in mapping
        assert mapping['bottle'] == 'medicine_bottle'
        assert 'scissors' in mapping
        assert mapping['scissors'] == 'surgical_instrument'

class TestAnnotationValidator:
    """Test annotation validation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = AnnotationValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        rules = self.validator.validation_rules
        
        assert 'min_bbox_size' in rules
        assert 'max_bbox_size' in rules
        assert 'max_objects_per_image' in rules
        assert 'valid_class_ids' in rules
        
        assert rules['min_bbox_size'] == 0.01
        assert rules['max_bbox_size'] == 0.9
        assert len(rules['valid_class_ids']) == 10
    
    def test_validate_valid_annotation(self):
        """Test validation of valid annotation"""
        # Create test files
        img_path = self.temp_dir + "/test.jpg"
        ann_path = self.temp_dir + "/test.txt"
        
        Image.new('RGB', (640, 480)).save(img_path)
        with open(ann_path, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("1 0.3 0.7 0.1 0.2\n")
        
        result = self.validator.validate_annotation_file(ann_path, img_path)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert result['object_count'] == 2
    
    def test_validate_invalid_format(self):
        """Test validation of invalid annotation format"""
        # Create test files
        img_path = self.temp_dir + "/test.jpg"
        ann_path = self.temp_dir + "/test.txt"
        
        Image.new('RGB', (640, 480)).save(img_path)
        with open(ann_path, 'w') as f:
            f.write("0 0.5 0.5\n")  # Missing width and height
        
        result = self.validator.validate_annotation_file(ann_path, img_path)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert "Invalid format" in result['errors'][0]
    
    def test_validate_invalid_class_id(self):
        """Test validation of invalid class ID"""
        # Create test files
        img_path = self.temp_dir + "/test.jpg"
        ann_path = self.temp_dir + "/test.txt"
        
        Image.new('RGB', (640, 480)).save(img_path)
        with open(ann_path, 'w') as f:
            f.write("15 0.5 0.5 0.2 0.3\n")  # Invalid class ID (>9)
        
        result = self.validator.validate_annotation_file(ann_path, img_path)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert "Invalid class ID" in result['errors'][0]

class TestDataQualityChecker:
    """Test data quality checking functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            (Path(self.temp_dir) / 'images' / split).mkdir(parents=True)
            (Path(self.temp_dir) / 'labels' / split).mkdir(parents=True)
        
        self.checker = DataQualityChecker(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checker_initialization(self):
        """Test quality checker initialization"""
        assert self.checker.dataset_path.exists()
        assert isinstance(self.checker.quality_report, dict)
    
    def test_check_image_quality_empty_dataset(self):
        """Test image quality check on empty dataset"""
        stats = self.checker.check_image_quality()
        
        assert stats['total_images'] == 0
        assert len(stats['corrupted_images']) == 0
        assert len(stats['low_quality_images']) == 0
    
    def test_check_image_quality_with_images(self):
        """Test image quality check with sample images"""
        # Create sample images
        train_dir = Path(self.temp_dir) / 'images' / 'train'
        
        # Good quality image
        good_img = Image.new('RGB', (640, 480), 'red')
        good_img.save(train_dir / 'good.jpg')
        
        # Small image (should be flagged)
        small_img = Image.new('RGB', (100, 100), 'blue')
        small_img.save(train_dir / 'small.jpg')
        
        stats = self.checker.check_image_quality()
        
        assert stats['total_images'] == 2
        assert len(stats['low_quality_images']) >= 1  # Small image should be flagged
    
    def test_check_annotation_quality_empty_dataset(self):
        """Test annotation quality check on empty dataset"""
        stats = self.checker.check_annotation_quality()
        
        assert stats['total_annotations'] == 0
        assert len(stats['missing_annotations']) == 0
        assert len(stats['invalid_annotations']) == 0
    
    def test_check_annotation_quality_with_data(self):
        """Test annotation quality check with sample data"""
        # Create sample data
        images_dir = Path(self.temp_dir) / 'images' / 'train'
        labels_dir = Path(self.temp_dir) / 'labels' / 'train'
        
        # Good image with annotation
        Image.new('RGB', (640, 480)).save(images_dir / 'good.jpg')
        with open(labels_dir / 'good.txt', 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
        
        # Image without annotation
        Image.new('RGB', (640, 480)).save(images_dir / 'missing.jpg')
        
        # Image with invalid annotation
        Image.new('RGB', (640, 480)).save(images_dir / 'invalid.jpg')
        with open(labels_dir / 'invalid.txt', 'w') as f:
            f.write("0 0.5 0.5\n")  # Missing values
        
        stats = self.checker.check_annotation_quality()
        
        assert stats['total_annotations'] >= 1
        assert len(stats['missing_annotations']) >= 1  # missing.jpg
        assert len(stats['invalid_annotations']) >= 1  # invalid.txt
    
    def test_check_dataset_balance(self):
        """Test dataset balance checking"""
        # Create sample annotations with class imbalance
        labels_dir = Path(self.temp_dir) / 'labels' / 'train'
        
        # Many syringes
        for i in range(5):
            with open(labels_dir / f'syringe_{i}.txt', 'w') as f:
                f.write("0 0.5 0.5 0.2 0.3\n")
        
        # Few bandages
        with open(labels_dir / 'bandage.txt', 'w') as f:
            f.write("1 0.5 0.5 0.2 0.3\n")
        
        stats = self.checker.check_dataset_balance()
        
        assert len(stats['class_counts']) >= 2
        assert stats['imbalance_ratio'] > 1  # Should detect imbalance
        if stats['imbalance_ratio'] > 10:
            assert len(stats['recommendations']) > 0

class TestIntegration:
    """Integration tests for dataset components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_dataset_workflow(self):
        """Test complete dataset preparation workflow"""
        # 1. Initialize dataset
        dataset = MedicalInventoryDataset(self.temp_dir)
        
        # 2. Create sample data
        images_dir = dataset.images_dir
        labels_dir = dataset.labels_dir
        
        for i in range(10):
            img_path = images_dir / f"sample_{i}.jpg"
            label_path = labels_dir / f"sample_{i}.txt"
            
            # Create image
            Image.new('RGB', (640, 480), (i*25, i*25, i*25)).save(img_path)
            
            # Create annotation
            with open(label_path, 'w') as f:
                class_id = i % 3  # Rotate between classes 0, 1, 2
                f.write(f"{class_id} 0.5 0.5 0.2 0.3\n")
        
        # 3. Split dataset
        dataset.split_dataset()
        
        # 4. Generate statistics
        stats = dataset.generate_statistics()
        
        # 5. Quality check
        checker = DataQualityChecker(self.temp_dir)
        quality_report = checker.generate_quality_report()
        
        # 6. Create dataset config
        dataset.create_dataset_yaml()
        
        # Verify everything worked
        assert stats['total_images'] == 10
        assert (dataset.dataset_root / "dataset.yaml").exists()
        assert quality_report['image_quality']['stats']['total_images'] == 10
        assert len(quality_report['recommendations']) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])