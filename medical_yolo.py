#!/usr/bin/env python3
"""
Custom YOLO Architecture for Medical Inventory Detection
Enhanced architecture optimized for small medical objects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math
from pathlib import Path
import yaml

class CBR(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attention))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions"""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CBR(c1, 2 * self.c, 1, 1)
        self.cv2 = CBR((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBR(c1, c_, k[0], 1)
        self.cv2 = CBR(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPP(nn.Module):
    """Spatial Pyramid Pooling layer"""
    def __init__(self, c1: int, c2: int, k: Tuple = (5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBR(c1, c_, 1, 1)
        self.cv2 = CBR(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class MedicalYOLOBackbone(nn.Module):
    """Enhanced backbone for medical object detection"""
    
    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0):
        super().__init__()
        
        # Calculate channel dimensions
        def make_divisible(x, divisor=8):
            return int(math.ceil(x / divisor) * divisor)
        
        c1 = make_divisible(64 * width_mult)
        c2 = make_divisible(128 * width_mult)
        c3 = make_divisible(256 * width_mult)
        c4 = make_divisible(512 * width_mult)
        c5 = make_divisible(1024 * width_mult)
        
        # Calculate depth
        d1 = max(round(3 * depth_mult), 1)
        d2 = max(round(6 * depth_mult), 1)
        d3 = max(round(9 * depth_mult), 1)
        d4 = max(round(3 * depth_mult), 1)
        
        # Stem
        self.stem = nn.Sequential(
            CBR(3, c1//2, 3, 2, 1),  # P1/2
            CBR(c1//2, c1, 3, 2, 1),  # P2/4
        )
        
        # Stage 1 - Enhanced for small objects
        self.stage1 = nn.Sequential(
            C2f(c1, c2, d1, True),
            CBR(c2, c2, 3, 2, 1),  # P3/8
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            C2f(c2, c3, d2, True),
            CBR(c3, c3, 3, 2, 1),  # P4/16
        )
        
        # Stage 3 
        self.stage3 = nn.Sequential(
            C2f(c3, c4, d3, True),
            CBR(c4, c4, 3, 2, 1),  # P5/32
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            C2f(c4, c5, d4, True),
            SPP(c5, c5, (5, 9, 13)),
        )
        
        # Attention modules for medical objects
        self.attention_p3 = CBAM(c2)
        self.attention_p4 = CBAM(c3) 
        self.attention_p5 = CBAM(c4)
        self.attention_p6 = CBAM(c5)
        
    def forward(self, x):
        """Forward pass returning multi-scale features"""
        x = self.stem(x)  # P2
        
        p3 = self.stage1(x)  # P3/8
        p3 = self.attention_p3(p3)
        
        p4 = self.stage2(p3)  # P4/16
        p4 = self.attention_p4(p4)
        
        p5 = self.stage3(p4)  # P5/32
        p5 = self.attention_p5(p5)
        
        p6 = self.stage4(p5)  # P6/32 (no stride increase, just processing)
        p6 = self.attention_p6(p6)
        
        return p3, p4, p5, p6

class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network for better feature fusion"""
    
    def __init__(self, channels: List[int], out_channels: int = 256, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            CBR(c, out_channels, 1) for c in channels
        ])
        
        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, len(channels)) for _ in range(num_layers)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Apply lateral connections
        laterals = [lateral(feat) for lateral, feat in zip(self.lateral_convs, features)]
        
        # Apply BiFPN layers
        for bifpn in self.bifpn_layers:
            laterals = bifpn(laterals)
        
        return laterals

class BiFPNLayer(nn.Module):
    """Single BiFPN layer with weighted feature fusion"""
    
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        # Upsampling weights
        self.up_weights = nn.Parameter(torch.ones(num_levels - 1, 2))
        # Downsampling weights  
        self.down_weights = nn.Parameter(torch.ones(num_levels - 1, 3))
        
        # Convolutions for fusion
        self.up_convs = nn.ModuleList([
            CBR(channels, channels, 3, 1, 1) for _ in range(num_levels - 1)
        ])
        self.down_convs = nn.ModuleList([
            CBR(channels, channels, 3, 1, 1) for _ in range(num_levels - 1)
        ])
        
        self.eps = 1e-4
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Top-down pathway
        up_features = [features[-1]]  # Start with highest level
        
        for i in range(len(features) - 2, -1, -1):
            # Weighted fusion
            w = F.relu(self.up_weights[i])
            w = w / (w.sum() + self.eps)
            
            upsampled = F.interpolate(up_features[-1], scale_factor=2, mode='nearest')
            fused = w[0] * features[i] + w[1] * upsampled
            up_features.append(self.up_convs[i](fused))
        
        up_features.reverse()
        
        # Bottom-up pathway
        out_features = [up_features[0]]
        
        for i in range(1, len(features)):
            # Weighted fusion
            w = F.relu(self.down_weights[i-1])
            w = w / (w.sum() + self.eps)
            
            downsampled = F.max_pool2d(out_features[-1], kernel_size=3, stride=2, padding=1)
            fused = w[0] * up_features[i] + w[1] * features[i] + w[2] * downsampled
            out_features.append(self.down_convs[i-1](fused))
        
        return out_features

class MedicalYOLOHead(nn.Module):
    """Enhanced detection head with medical object focus"""
    
    def __init__(self, nc: int = 10, anchors: int = 3, ch: List[int] = [256, 256, 256, 256]):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(ch)  # number of detection layers
        self.na = anchors  # number of anchors
        
        # Detection layers
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
        # Class-specific branches for medical items
        self.class_branches = nn.ModuleList([
            nn.Sequential(
                CBR(x, x//2, 3, 1, 1),
                nn.Conv2d(x//2, self.nc * self.na, 1)
            ) for x in ch
        ])
        
        # Confidence calibration
        self.conf_calibration = nn.ModuleList([
            nn.Sequential(
                CBR(x, x//4, 3, 1, 1),
                nn.Conv2d(x//4, self.na, 1),
                nn.Sigmoid()
            ) for x in ch
        ])
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        
        for i, xi in enumerate(x):
            # Standard detection output
            det_out = self.m[i](xi)
            
            # Class-specific output
            cls_out = self.class_branches[i](xi)
            
            # Confidence calibration
            conf_cal = self.conf_calibration[i](xi)
            
            # Reshape outputs
            bs, _, ny, nx = xi.shape
            det_out = det_out.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            cls_out = cls_out.view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            conf_cal = conf_cal.view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # Apply confidence calibration
            det_out[..., 4:5] = det_out[..., 4:5] * conf_cal
            det_out[..., 5:] = cls_out
            
            outputs.append(det_out)
        
        return outputs

class MedicalYOLO(nn.Module):
    """Complete Medical YOLO model"""
    
    def __init__(self, nc: int = 10, width_mult: float = 1.0, depth_mult: float = 1.0):
        super().__init__()
        self.nc = nc
        
        # Backbone
        self.backbone = MedicalYOLOBackbone(width_mult, depth_mult)
        
        # Get backbone output channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            backbone_outputs = self.backbone(dummy_input)
            backbone_channels = [x.shape[1] for x in backbone_outputs]
        
        # Neck (BiFPN)
        self.neck = BiFPN(backbone_channels, out_channels=256, num_layers=2)
        
        # Head
        self.head = MedicalYOLOHead(nc=nc, ch=[256] * len(backbone_channels))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass"""
        # Backbone
        backbone_features = self.backbone(x)
        
        # Neck
        neck_features = self.neck(backbone_features)
        
        # Head
        outputs = self.head(neck_features)
        
        return outputs

def create_medical_yolo_variants():
    """Create different variants of Medical YOLO"""
    variants = {
        'medical_nano': {'width_mult': 0.25, 'depth_mult': 0.33},
        'medical_small': {'width_mult': 0.5, 'depth_mult': 0.33},
        'medical_medium': {'width_mult': 0.75, 'depth_mult': 0.67},
        'medical_large': {'width_mult': 1.0, 'depth_mult': 1.0},
        'medical_xlarge': {'width_mult': 1.25, 'depth_mult': 1.33}
    }
    
    models = {}
    for name, params in variants.items():
        models[name] = MedicalYOLO(nc=10, **params)
    
    return models

def save_model_config(model_name: str, save_path: str = "./configs/"):
    """Save model configuration"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    variants = {
        'medical_nano': {'width_mult': 0.25, 'depth_mult': 0.33, 'params': '2M'},
        'medical_small': {'width_mult': 0.5, 'depth_mult': 0.33, 'params': '7M'},
        'medical_medium': {'width_mult': 0.75, 'depth_mult': 0.67, 'params': '25M'},
        'medical_large': {'width_mult': 1.0, 'depth_mult': 1.0, 'params': '50M'},
        'medical_xlarge': {'width_mult': 1.25, 'depth_mult': 1.33, 'params': '100M'}
    }
    
    config = {
        'model': model_name,
        'nc': 10,
        'names': [
            'syringe', 'bandage', 'medicine_bottle', 'pills_blister',
            'surgical_instrument', 'gloves_box', 'mask', 'iv_bag',
            'thermometer', 'first_aid_supply'
        ],
        'architecture': variants.get(model_name, variants['medical_medium'])
    }
    
    with open(f"{save_path}/{model_name}.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    """Test model creation"""
    print("Creating Medical YOLO variants...")
    
    models = create_medical_yolo_variants()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        try:
            with torch.no_grad():
                outputs = model(dummy_input)
            print(f"  ✓ Forward pass successful")
            print(f"  ✓ Output shapes: {[out.shape for out in outputs]}")
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Parameters: {params:,}")
            
            # Save config
            save_model_config(name)
            print(f"  ✓ Config saved")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    main()