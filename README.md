---
title: Medical Inventory Checker
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Medical Inventory Checker 🏥

An advanced AI-powered application for checking medical case inventory using computer vision, multimodal models, and DINO self-supervised learning. Designed for medical equipment cases containing screws, tools, and instruments.

## ✨ Features

- **🌐 Bilingual Interface**: Full support for English and Spanish with dynamic language switching
- **🔍 DINO Enhanced Detection**: Self-supervised vision transformers for superior change detection  
- **🟢 Green Bounding Box Visualization**: Professional medical-grade visual annotations
- **🤖 Multiple AI Models**: Qwen2.5-VL, Kosmos-2, Florence-2, and DINO integration
- **📊 Detailed Analysis Reports**: Comprehensive bilingual reports with visual statistics
- **⚡ Enhanced Annotations**: Lightning bolt indicators for DINO-enhanced detections

## 🚀 How to Use

1. **Select Language**: Choose English or Español from the language selector
2. **Upload Images**: Add "before" and "after" images of your medical case
3. **Choose Model**: Select AI model (Qwen2.5-VL recommended for best results)
4. **Set Threshold**: Adjust detection sensitivity (0.5 default)
5. **Analyze**: Click "Check Inventory" / "Verificar Inventario" 
6. **Review Results**: View green-boxed annotations and detailed bilingual reports

## 🧠 AI Models

- **Qwen2.5-VL-3B**: Advanced vision-language model for precise object detection
- **Kosmos-2**: Microsoft's grounding model for accurate spatial localization  
- **Florence-2**: Lightweight baseline for fast processing
- **DINO ViT**: Self-supervised vision transformer for change region detection

## Technical Details

This application uses Hugging Face Transformers to run vision-language models that can detect and compare objects in medical inventory cases. The system highlights differences between before and after states to help medical staff quickly identify missing or changed items.

## Limitations

- Best results with clear, well-lit images
- Works optimally with organized medical cases
- May require fine-tuning for specific medical equipment types

## About

Created for medical professionals to streamline inventory management and reduce errors in medical equipment tracking.