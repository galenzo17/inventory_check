#!/usr/bin/env python3
"""
YOLO Object Detection, Segmentation and Counting App
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from ultralytics import YOLO
import spaces
import colorsys
import io

class YOLODetectionApp:
    def __init__(self):
        self.models = {
            "YOLOv8n": "yolov8n.pt",
            "YOLOv8s": "yolov8s.pt", 
            "YOLOv8m": "yolov8m.pt",
            "YOLOv8l": "yolov8l.pt",
            "YOLOv8x": "yolov8x.pt",
            "YOLOv8n-seg": "yolov8n-seg.pt",
            "YOLOv8s-seg": "yolov8s-seg.pt",
            "YOLOv8m-seg": "yolov8m-seg.pt",
            "YOLOv8l-seg": "yolov8l-seg.pt",
            "YOLOv8x-seg": "yolov8x-seg.pt"
        }
        self.current_model = None
        self.current_model_name = None
        
    def load_model(self, model_name):
        """Load YOLO model if not already loaded"""
        if self.current_model_name != model_name:
            try:
                model_path = self.models[model_name]
                self.current_model = YOLO(model_path)
                self.current_model_name = model_name
                return True
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return False
        return True
    
    def get_distinct_colors(self, n):
        """Generate visually distinct colors for each class"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.9
            lightness = 0.5
            rgb = colorsys.hsv_to_rgb(hue, saturation, lightness)
            colors.append(tuple(int(255 * x) for x in rgb))
        return colors
    
    @spaces.GPU(duration=60)
    def detect_objects(self, image, model_name, confidence_threshold=0.25, iou_threshold=0.45, 
                      show_labels=True, show_confidence=True, show_boxes=True):
        """Perform object detection on the image"""
        
        if image is None:
            return None, "Please upload an image first."
        
        # Load model
        if not self.load_model(model_name):
            return None, f"Failed to load model: {model_name}"
        
        try:
            # Run inference
            results = self.current_model(image, conf=confidence_threshold, iou=iou_threshold)
            
            # Process results
            result = results[0]
            
            # Convert image to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Create a copy for drawing
            annotated_img = img_array.copy()
            
            # Get class names and colors
            names = result.names
            num_classes = len(names)
            colors = self.get_distinct_colors(num_classes)
            
            # Count objects by class
            class_counts = {}
            
            # Draw bounding boxes
            if result.boxes is not None and show_boxes:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = names[cls]
                    
                    # Update count
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Get color for this class
                    color = colors[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    label_parts = []
                    if show_labels:
                        label_parts.append(class_name)
                    if show_confidence:
                        label_parts.append(f"{conf:.2f}")
                    
                    if label_parts:
                        label = " ".join(label_parts)
                        
                        # Draw label background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(annotated_img, 
                                    (x1, y1 - label_size[1] - 4),
                                    (x1 + label_size[0], y1),
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(annotated_img, label,
                                  (x1, y1 - 2),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 255, 255), 1)
            
            # Convert back to PIL Image
            result_image = Image.fromarray(annotated_img)
            
            # Generate report
            total_objects = sum(class_counts.values())
            report = f"## Detection Results\n\n"
            report += f"**Total Objects Detected:** {total_objects}\n\n"
            
            if class_counts:
                report += "### Object Counts by Class:\n"
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    report += f"- **{class_name}:** {count} object{'s' if count > 1 else ''}\n"
            else:
                report += "No objects detected with the current threshold settings.\n"
            
            report += f"\n### Detection Settings:\n"
            report += f"- Model: {model_name}\n"
            report += f"- Confidence Threshold: {confidence_threshold:.2f}\n"
            report += f"- IoU Threshold: {iou_threshold:.2f}\n"
            
            return result_image, report
            
        except Exception as e:
            return None, f"Error during detection: {str(e)}"
    
    @spaces.GPU(duration=60)
    def segment_objects(self, image, model_name, confidence_threshold=0.25, iou_threshold=0.45,
                       show_masks=True, show_boxes=False, show_labels=True):
        """Perform instance segmentation on the image"""
        
        if image is None:
            return None, "Please upload an image first."
        
        # Check if model supports segmentation
        if "-seg" not in model_name:
            return None, "Please select a segmentation model (with '-seg' suffix)"
        
        # Load model
        if not self.load_model(model_name):
            return None, f"Failed to load model: {model_name}"
        
        try:
            # Run inference
            results = self.current_model(image, conf=confidence_threshold, iou=iou_threshold)
            
            # Process results
            result = results[0]
            
            # Convert image to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Create a copy for drawing
            annotated_img = img_array.copy()
            overlay = np.zeros_like(annotated_img)
            
            # Get class names and colors
            names = result.names
            num_classes = len(names)
            colors = self.get_distinct_colors(num_classes)
            
            # Count objects by class
            class_counts = {}
            
            # Process masks and boxes
            if result.masks is not None and show_masks:
                masks = result.masks.data.cpu().numpy()
                
                for i, mask in enumerate(masks):
                    # Get class and confidence
                    cls = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    class_name = names[cls]
                    
                    # Update count
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Get color for this class
                    color = colors[cls]
                    
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
                    
                    # Apply mask with transparency
                    mask_bool = mask_resized > 0.5
                    overlay[mask_bool] = color
                    
                    # Draw contours
                    contours, _ = cv2.findContours((mask_resized > 0.5).astype(np.uint8), 
                                                  cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_img, contours, -1, color, 2)
                    
                    # Add label if requested
                    if show_labels and len(contours) > 0:
                        # Get centroid of largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            label = f"{class_name} {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            
                            # Draw label background
                            cv2.rectangle(annotated_img,
                                        (cx - label_size[0]//2, cy - label_size[1]//2 - 2),
                                        (cx + label_size[0]//2, cy + label_size[1]//2 + 2),
                                        color, -1)
                            
                            # Draw label text
                            cv2.putText(annotated_img, label,
                                      (cx - label_size[0]//2, cy + label_size[1]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (255, 255, 255), 1)
            
            # Draw boxes if requested
            if result.boxes is not None and show_boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    color = colors[cls]
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Blend overlay with original image
            if show_masks:
                annotated_img = cv2.addWeighted(annotated_img, 0.7, overlay, 0.3, 0)
            
            # Convert back to PIL Image
            result_image = Image.fromarray(annotated_img)
            
            # Generate report
            total_objects = sum(class_counts.values())
            report = f"## Segmentation Results\n\n"
            report += f"**Total Objects Segmented:** {total_objects}\n\n"
            
            if class_counts:
                report += "### Object Counts by Class:\n"
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    report += f"- **{class_name}:** {count} instance{'s' if count > 1 else ''}\n"
            else:
                report += "No objects detected with the current threshold settings.\n"
            
            report += f"\n### Segmentation Settings:\n"
            report += f"- Model: {model_name}\n"
            report += f"- Confidence Threshold: {confidence_threshold:.2f}\n"
            report += f"- IoU Threshold: {iou_threshold:.2f}\n"
            report += f"- Show Masks: {show_masks}\n"
            report += f"- Show Boxes: {show_boxes}\n"
            
            return result_image, report
            
        except Exception as e:
            return None, f"Error during segmentation: {str(e)}"
    
    def create_interface(self):
        """Create the YOLO interface components"""
        
        with gr.Column():
            gr.HTML("""
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); border-radius: 15px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0; font-size: 1.8em;">🎯 YOLO Object Detection & Segmentation</h2>
                <p style="color: white; margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1em;">Real-time object detection, segmentation, and counting</p>
            </div>
            """)
            
            with gr.Tabs():
                # Object Detection Tab
                with gr.Tab("🔍 Object Detection"):
                    gr.Markdown("""
                    ### Instructions:
                    1. Upload an image for analysis
                    2. Select a YOLO model (larger models are more accurate but slower)
                    3. Adjust confidence and IoU thresholds
                    4. Click 'Detect Objects' to analyze
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            detect_image = gr.Image(
                                label="Upload Image",
                                type="pil",
                                height=400
                            )
                            
                            detect_model = gr.Dropdown(
                                choices=["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
                                value="YOLOv8n",
                                label="Select YOLO Model"
                            )
                            
                            with gr.Row():
                                detect_conf = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.25,
                                    step=0.05,
                                    label="Confidence Threshold"
                                )
                                
                                detect_iou = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.45,
                                    step=0.05,
                                    label="IoU Threshold"
                                )
                            
                            with gr.Row():
                                show_labels = gr.Checkbox(value=True, label="Show Labels")
                                show_conf = gr.Checkbox(value=True, label="Show Confidence")
                                show_boxes = gr.Checkbox(value=True, label="Show Boxes")
                            
                            detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            detect_output = gr.Image(
                                label="Detection Results",
                                type="pil",
                                height=400
                            )
                            
                            detect_report = gr.Markdown(
                                value="Upload an image and click 'Detect Objects' to start"
                            )
                    
                    # Connect detection function
                    detect_btn.click(
                        fn=self.detect_objects,
                        inputs=[detect_image, detect_model, detect_conf, detect_iou, 
                               show_labels, show_conf, show_boxes],
                        outputs=[detect_output, detect_report]
                    )
                
                # Segmentation Tab
                with gr.Tab("🎨 Instance Segmentation"):
                    gr.Markdown("""
                    ### Instructions:
                    1. Upload an image for segmentation
                    2. Select a YOLO segmentation model (with '-seg' suffix)
                    3. Adjust detection parameters
                    4. Click 'Segment Objects' to analyze
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            seg_image = gr.Image(
                                label="Upload Image",
                                type="pil",
                                height=400
                            )
                            
                            seg_model = gr.Dropdown(
                                choices=["YOLOv8n-seg", "YOLOv8s-seg", "YOLOv8m-seg", 
                                        "YOLOv8l-seg", "YOLOv8x-seg"],
                                value="YOLOv8n-seg",
                                label="Select Segmentation Model"
                            )
                            
                            with gr.Row():
                                seg_conf = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.25,
                                    step=0.05,
                                    label="Confidence Threshold"
                                )
                                
                                seg_iou = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.45,
                                    step=0.05,
                                    label="IoU Threshold"
                                )
                            
                            with gr.Row():
                                show_masks = gr.Checkbox(value=True, label="Show Masks")
                                show_seg_boxes = gr.Checkbox(value=False, label="Show Boxes")
                                show_seg_labels = gr.Checkbox(value=True, label="Show Labels")
                            
                            seg_btn = gr.Button("🎨 Segment Objects", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            seg_output = gr.Image(
                                label="Segmentation Results",
                                type="pil",
                                height=400
                            )
                            
                            seg_report = gr.Markdown(
                                value="Upload an image and click 'Segment Objects' to start"
                            )
                    
                    # Connect segmentation function
                    seg_btn.click(
                        fn=self.segment_objects,
                        inputs=[seg_image, seg_model, seg_conf, seg_iou,
                               show_masks, show_seg_boxes, show_seg_labels],
                        outputs=[seg_output, seg_report]
                    )
            
                # Batch Processing Tab
                with gr.Tab("📦 Batch Processing"):
                    gr.Markdown("""
                    ### Instructions:
                    1. Upload multiple images for batch analysis
                    2. Select processing mode and model
                    3. Set detection parameters
                    4. Click 'Process Batch' to analyze all images
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_images = gr.File(
                                label="Upload Multiple Images",
                                file_count="multiple",
                                file_types=["image"]
                            )
                            
                            batch_mode = gr.Radio(
                                choices=["Detection", "Segmentation"],
                                value="Detection",
                                label="Processing Mode"
                            )
                            
                            batch_model = gr.Dropdown(
                                choices=["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x",
                                        "YOLOv8n-seg", "YOLOv8s-seg", "YOLOv8m-seg", 
                                        "YOLOv8l-seg", "YOLOv8x-seg"],
                                value="YOLOv8n",
                                label="Select Model"
                            )
                            
                            with gr.Row():
                                batch_conf = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.25,
                                    step=0.05,
                                    label="Confidence Threshold"
                                )
                                
                                batch_iou = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.45,
                                    step=0.05,
                                    label="IoU Threshold"
                                )
                            
                            batch_btn = gr.Button("📦 Process Batch", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            batch_gallery = gr.Gallery(
                                label="Batch Results",
                                show_label=True,
                                elem_id="batch_gallery",
                                columns=2,
                                rows=2,
                                height=400
                            )
                            
                            batch_report = gr.Markdown(
                                value="Upload images and click 'Process Batch' to start"
                            )
                    
                    # Connect batch processing function
                    batch_btn.click(
                        fn=self.process_batch,
                        inputs=[batch_images, batch_mode, batch_model, batch_conf, batch_iou],
                        outputs=[batch_gallery, batch_report]
                    )
            
            # Add examples section
            gr.Markdown("### 📸 Example Images")
            gr.Markdown("Try these example images to test the detection and segmentation capabilities:")
    
    @spaces.GPU(duration=120)
    def process_batch(self, files, mode, model_name, confidence_threshold=0.25, iou_threshold=0.45):
        """Process multiple images in batch"""
        
        if not files:
            return None, "Please upload at least one image."
        
        # Load model
        if not self.load_model(model_name):
            return None, f"Failed to load model: {model_name}"
        
        try:
            processed_images = []
            total_stats = {}
            
            for file in files:
                # Load image
                image = Image.open(file.name)
                
                # Process based on mode
                if mode == "Detection":
                    result_img, _ = self.detect_objects(
                        image, model_name, confidence_threshold, iou_threshold,
                        show_labels=True, show_confidence=True, show_boxes=True
                    )
                else:  # Segmentation
                    result_img, _ = self.segment_objects(
                        image, model_name, confidence_threshold, iou_threshold,
                        show_masks=True, show_boxes=False, show_labels=True
                    )
                
                if result_img:
                    processed_images.append(result_img)
                    
                    # Run inference for statistics
                    results = self.current_model(image, conf=confidence_threshold, iou=iou_threshold)
                    result = results[0]
                    
                    # Count objects
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            class_name = result.names[cls]
                            total_stats[class_name] = total_stats.get(class_name, 0) + 1
            
            # Generate batch report
            report = f"## Batch Processing Results\n\n"
            report += f"**Images Processed:** {len(processed_images)}/{len(files)}\n\n"
            
            if total_stats:
                report += "### Total Object Counts Across All Images:\n"
                for class_name, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
                    report += f"- **{class_name}:** {count} object{'s' if count > 1 else ''}\n"
            else:
                report += "No objects detected in the batch.\n"
            
            report += f"\n### Processing Settings:\n"
            report += f"- Mode: {mode}\n"
            report += f"- Model: {model_name}\n"
            report += f"- Confidence Threshold: {confidence_threshold:.2f}\n"
            report += f"- IoU Threshold: {iou_threshold:.2f}\n"
            
            return processed_images, report
            
        except Exception as e:
            return None, f"Error during batch processing: {str(e)}"