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

An AI-powered application for checking medical case inventory using computer vision and multimodal models. Designed for medical equipment cases containing screws, tools, and instruments.

## Features

- **Visual Inventory Comparison**: Compare before/after images of medical cases
- **AI-Powered Detection**: Uses state-of-the-art vision models
- **Difference Highlighting**: Visually marks differences between inventory states
- **Multiple Model Support**: Test different vision models for optimal accuracy

## How to Use

1. Upload a "before" image of your medical case
2. Upload an "after" image of the medical case
3. Select the AI model to use (Qwen2.5-VL recommended)
4. Click "Check Inventory" to analyze differences
5. View the results with highlighted changes

## Supported Models

- **Qwen2.5-VL-3B**: Fast and accurate for object detection
- **Kosmos-2**: Microsoft's grounding model for precise localization
- **Florence-2**: Lightweight baseline model

## Technical Details

This application uses Hugging Face Transformers to run vision-language models that can detect and compare objects in medical inventory cases. The system highlights differences between before and after states to help medical staff quickly identify missing or changed items.

## Limitations

- Best results with clear, well-lit images
- Works optimally with organized medical cases
- May require fine-tuning for specific medical equipment types

## About

Created for medical professionals to streamline inventory management and reduce errors in medical equipment tracking.