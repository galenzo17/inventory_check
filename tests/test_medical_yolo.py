#!/usr/bin/env python3
"""
Comprehensive tests for Medical YOLO models
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from medical_yolo import (
    MedicalYOLO, 
    MedicalYOLOBackbone, 
    BiFPN, 
    MedicalYOLOHead,
    create_medical_yolo_variants,
    CBAM
)

class TestMedicalYOLOBackbone:
    """Test Medical YOLO backbone architecture"""
    
    def test_backbone_initialization(self):
        """Test backbone initialization with different multipliers"""
        backbone = MedicalYOLOBackbone(width_mult=1.0, depth_mult=1.0)
        assert backbone is not None
        
        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        outputs = backbone(x)
        
        assert len(outputs) == 4  # P3, P4, P5, P6
        assert all(isinstance(out, torch.Tensor) for out in outputs)
    
    def test_backbone_different_sizes(self):
        """Test backbone with different size multipliers"""
        sizes = [
            (0.25, 0.33),  # nano
            (0.5, 0.33),   # small
            (1.0, 1.0),    # medium
            (1.25, 1.33)   # xlarge
        ]
        
        for width_mult, depth_mult in sizes:
            backbone = MedicalYOLOBackbone(width_mult=width_mult, depth_mult=depth_mult)
            x = torch.randn(1, 3, 320, 320)
            outputs = backbone(x)
            
            assert len(outputs) == 4
            # Check output dimensions are reasonable
            for i, out in enumerate(outputs):
                assert out.dim() == 4  # NCHW format
                assert out.size(0) == 1  # batch size
    
    def test_cbam_attention(self):
        """Test CBAM attention mechanism"""
        cbam = CBAM(256, reduction_ratio=16)
        x = torch.randn(2, 256, 32, 32)
        
        output = cbam(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should modify input

class TestBiFPN:
    """Test Bidirectional Feature Pyramid Network"""
    
    def test_bifpn_initialization(self):
        """Test BiFPN initialization"""
        channels = [128, 256, 512, 1024]
        bifpn = BiFPN(channels, out_channels=256, num_layers=2)
        
        assert bifpn.num_layers == 2
        assert len(bifpn.lateral_convs) == len(channels)
    
    def test_bifpn_forward(self):
        """Test BiFPN forward pass"""
        channels = [128, 256, 512, 1024]
        bifpn = BiFPN(channels, out_channels=256, num_layers=2)
        
        # Create dummy features
        features = [
            torch.randn(1, c, 80, 80) for c in channels
        ]
        
        output = bifpn(features)
        
        assert len(output) == len(features)
        assert all(out.size(1) == 256 for out in output)  # All have out_channels

class TestMedicalYOLOHead:
    """Test Medical YOLO detection head"""
    
    def test_head_initialization(self):
        """Test head initialization"""
        head = MedicalYOLOHead(nc=10, anchors=3, ch=[256, 256, 256, 256])
        
        assert head.nc == 10
        assert head.na == 3
        assert head.nl == 4
    
    def test_head_forward(self):
        """Test head forward pass"""
        head = MedicalYOLOHead(nc=10, anchors=3, ch=[256, 256, 256, 256])
        
        # Create dummy feature maps
        features = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 256, 20, 20),
            torch.randn(1, 256, 10, 10)
        ]
        
        outputs = head(features)
        
        assert len(outputs) == 4
        for i, out in enumerate(outputs):
            expected_h = features[i].size(2)
            expected_w = features[i].size(3)
            assert out.shape == (1, 3, expected_h, expected_w, 15)  # nc + 5

class TestMedicalYOLO:
    """Test complete Medical YOLO model"""
    
    def test_model_creation(self):
        """Test model creation and initialization"""
        model = MedicalYOLO(nc=10, width_mult=1.0, depth_mult=1.0)
        
        assert model.nc == 10
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'head')
    
    def test_model_forward(self):
        """Test model forward pass"""
        model = MedicalYOLO(nc=10, width_mult=0.5, depth_mult=0.33)  # Small model for testing
        model.eval()
        
        x = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(x)
        
        assert len(outputs) == 4
        assert all(isinstance(out, torch.Tensor) for out in outputs)
    
    def test_model_variants_creation(self):
        """Test creation of all model variants"""
        variants = create_medical_yolo_variants()
        
        expected_variants = ['medical_nano', 'medical_small', 'medical_medium', 
                           'medical_large', 'medical_xlarge']
        
        assert len(variants) == len(expected_variants)
        assert all(variant in variants for variant in expected_variants)
        
        # Test each variant can do forward pass
        x = torch.randn(1, 3, 320, 320)  # Smaller input for speed
        
        for name, model in variants.items():
            model.eval()
            with torch.no_grad():
                outputs = model(x)
            assert len(outputs) == 4, f"Failed for {name}"
    
    def test_model_parameter_count(self):
        """Test model parameter counts are reasonable"""
        variants = create_medical_yolo_variants()
        
        param_counts = {}
        for name, model in variants.items():
            param_count = sum(p.numel() for p in model.parameters())
            param_counts[name] = param_count
        
        # Nano should have fewer parameters than xlarge
        assert param_counts['medical_nano'] < param_counts['medical_medium']
        assert param_counts['medical_medium'] < param_counts['medical_xlarge']
        
        # Sanity check - no model should be empty or too large
        for name, count in param_counts.items():
            assert count > 100_000, f"{name} has too few parameters: {count}"
            assert count < 200_000_000, f"{name} has too many parameters: {count}"

class TestModelInvariance:
    """Test model invariance properties"""
    
    def test_batch_size_invariance(self):
        """Test model works with different batch sizes"""
        model = MedicalYOLO(nc=10, width_mult=0.25, depth_mult=0.33)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            x = torch.randn(bs, 3, 320, 320)
            with torch.no_grad():
                outputs = model(x)
            
            assert len(outputs) == 4
            for out in outputs:
                assert out.size(0) == bs, f"Failed for batch size {bs}"
    
    def test_input_size_flexibility(self):
        """Test model with different input sizes"""
        model = MedicalYOLO(nc=10, width_mult=0.25, depth_mult=0.33)
        model.eval()
        
        input_sizes = [320, 416, 512, 640]
        
        for size in input_sizes:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                outputs = model(x)
            
            assert len(outputs) == 4, f"Failed for input size {size}"
    
    @pytest.mark.parametrize("width_mult,depth_mult", [
        (0.25, 0.33),
        (0.5, 0.33),
        (1.0, 1.0),
        (1.25, 1.33)
    ])
    def test_model_deterministic(self, width_mult, depth_mult):
        """Test model produces deterministic outputs"""
        torch.manual_seed(42)
        model1 = MedicalYOLO(nc=10, width_mult=width_mult, depth_mult=depth_mult)
        
        torch.manual_seed(42)
        model2 = MedicalYOLO(nc=10, width_mult=width_mult, depth_mult=depth_mult)
        
        # Models should have identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
        
        # Should produce identical outputs
        x = torch.randn(1, 3, 320, 320)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        for o1, o2 in zip(out1, out2):
            assert torch.allclose(o1, o2)

class TestModelPerformance:
    """Test model performance characteristics"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_compatibility(self):
        """Test model works on GPU"""
        model = MedicalYOLO(nc=10, width_mult=0.5, depth_mult=0.33)
        model = model.cuda()
        model.eval()
        
        x = torch.randn(1, 3, 320, 320).cuda()
        
        with torch.no_grad():
            outputs = model(x)
        
        assert len(outputs) == 4
        for out in outputs:
            assert out.is_cuda
    
    def test_inference_speed_benchmark(self):
        """Benchmark inference speed"""
        import time
        
        model = MedicalYOLO(nc=10, width_mult=0.5, depth_mult=0.33)
        model.eval()
        
        x = torch.randn(1, 3, 320, 320)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                _ = model(x)
                times.append(time.time() - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        # Should be reasonably fast (>10 FPS on CPU for small model)
        assert fps > 5, f"Too slow: {fps:.1f} FPS"
        assert avg_time < 0.5, f"Too slow: {avg_time:.3f}s per inference"
    
    def test_memory_usage(self):
        """Test memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and run model
        model = MedicalYOLO(nc=10, width_mult=1.0, depth_mult=1.0)
        model.eval()
        
        x = torch.randn(4, 3, 640, 640)  # Larger batch
        
        with torch.no_grad():
            outputs = model(x)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (less than 2GB increase)
        assert memory_increase < 2000, f"Excessive memory usage: {memory_increase:.1f} MB"

class TestModelSerialization:
    """Test model saving and loading"""
    
    def test_model_state_dict_save_load(self):
        """Test saving and loading model state dict"""
        model1 = MedicalYOLO(nc=10, width_mult=0.5, depth_mult=0.33)
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            # Save model
            torch.save(model1.state_dict(), f.name)
            
            # Create new model and load weights
            model2 = MedicalYOLO(nc=10, width_mult=0.5, depth_mult=0.33)
            model2.load_state_dict(torch.load(f.name))
            
            # Test outputs are identical
            x = torch.randn(1, 3, 320, 320)
            
            model1.eval()
            model2.eval()
            
            with torch.no_data():
                out1 = model1(x)
                out2 = model2(x)
            
            for o1, o2 in zip(out1, out2):
                assert torch.allclose(o1, o2, atol=1e-6)
    
    def test_model_config_save_load(self):
        """Test saving and loading model configurations"""
        from medical_yolo import save_model_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            
            # Save config
            save_model_config('medical_medium', str(config_path))
            
            # Load config
            config_file = config_path / 'medical_medium.yaml'
            assert config_file.exists()
            
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            assert config['model'] == 'medical_medium'
            assert config['nc'] == 10
            assert 'names' in config
            assert len(config['names']) == 10

@pytest.fixture
def sample_medical_classes():
    """Sample medical inventory classes for testing"""
    return [
        'syringe', 'bandage', 'medicine_bottle', 'pills_blister',
        'surgical_instrument', 'gloves_box', 'mask', 'iv_bag',
        'thermometer', 'first_aid_supply'
    ]

class TestModelIntegration:
    """Integration tests with other components"""
    
    @patch('medical_yolo.torch.jit.load')
    def test_model_loading_from_checkpoint(self, mock_load):
        """Test loading model from checkpoint file"""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Test loading (mocked)
        result = torch.jit.load('dummy_path.pt')
        
        mock_load.assert_called_once_with('dummy_path.pt', map_location='cpu')
        assert result == mock_model
    
    def test_model_output_format_compatibility(self):
        """Test model output format is compatible with YOLO standards"""
        model = MedicalYOLO(nc=10, width_mult=0.5, depth_mult=0.33)
        model.eval()
        
        x = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(x)
        
        # Check output format
        assert len(outputs) == 4, "Should have 4 detection layers"
        
        for i, output in enumerate(outputs):
            batch_size, num_anchors, grid_h, grid_w, predictions = output.shape
            
            assert batch_size == 1
            assert num_anchors == 3
            assert predictions == 15  # 10 classes + 5 (x, y, w, h, conf)
            
            # Grid sizes should decrease
            if i > 0:
                prev_grid_size = outputs[i-1].shape[2] * outputs[i-1].shape[3]
                curr_grid_size = grid_h * grid_w
                assert curr_grid_size <= prev_grid_size, f"Grid size should decrease at layer {i}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])