import gradio as gr
import numpy as np
from PIL import Image
import json
import os
import sys
import spaces
from pathlib import Path

# Add src directory to Python path for HF Spaces compatibility
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import local modules - simplified approach for HF Spaces
try:
    # Try direct imports from current directory first
    from models.inventory_checker import InventoryChecker
    from utils.visualization import create_difference_visualization
    from data.inventory_db import InventoryDatabase
except ImportError:
    try:
        # Try relative imports
        from .models.inventory_checker import InventoryChecker
        from .utils.visualization import create_difference_visualization
        from .data.inventory_db import InventoryDatabase
    except ImportError:
        # Last resort: add each subdirectory to path
        models_dir = current_dir / "models"
        utils_dir = current_dir / "utils"  
        data_dir = current_dir / "data"
        
        for dir_path in [models_dir, utils_dir, data_dir]:
            if str(dir_path) not in sys.path:
                sys.path.insert(0, str(dir_path))
                
        from inventory_checker import InventoryChecker
        from visualization import create_difference_visualization
        from inventory_db import InventoryDatabase

class MedicalInventoryApp:
    def __init__(self):
        self.db = InventoryDatabase()
        self.available_models = [
            "microsoft/kosmos-2-patch14-224",
            "Salesforce/blip2-opt-2.7b",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct"
        ]
        self.checker = None
        
    @spaces.GPU(duration=60)
    def process_images(self, before_image, after_image, model_name, threshold=0.5):
        """Process before/after images and detect inventory differences"""
        
        if before_image is None or after_image is None:
            return None, "Please upload both before and after images"
        
        try:
            # Initialize or update the checker with selected model
            if self.checker is None or self.checker.model_name != model_name:
                self.checker = InventoryChecker(model_name)
            
            # Perform inventory comparison
            results = self.checker.compare_inventory(before_image, after_image, threshold)
            
            # Create visualization with highlighted differences
            visualization = create_difference_visualization(
                before_image, 
                after_image, 
                results['differences']
            )
            
            # Format results for display
            report = self._generate_report(results)
            
            return visualization, report
            
        except Exception as e:
            return None, f"Error processing images: {str(e)}"
    
    def _generate_report(self, results):
        """Generate a human-readable report of inventory differences"""
        
        report = "## Inventory Analysis Report\n\n"
        
        # Add DINO analysis if available
        if results.get('analysis', {}).get('enhanced_with_dino'):
            dino_sim = results['analysis']['dino_similarity']
            dino_change = results['analysis']['dino_change_detected']
            change_mag = results['analysis']['dino_change_magnitude']
            
            report += f"🔬 **Enhanced Analysis (DINO)**\n"
            report += f"- Overall Similarity: {dino_sim:.3f}\n"
            report += f"- Change Detected: {'Yes' if dino_change else 'No'}\n"
            report += f"- Change Magnitude: {change_mag:.3f}\n\n"
        
        if not results['differences']:
            report += "✅ No significant differences detected\n"
        else:
            report += f"⚠️ Found {len(results['differences'])} differences:\n\n"
            
            for i, diff in enumerate(results['differences'], 1):
                report += f"**{i}. {diff['type']}**\n"
                report += f"   - Item: {diff.get('item', 'Unknown')}\n"
                report += f"   - Location: {diff.get('location', 'N/A')}\n"
                report += f"   - Confidence: {diff.get('confidence', 0):.2%}\n\n"
        
        # Add inventory summary from database
        report += "\n## Expected Inventory:\n"
        inventory = self.db.get_all_items()
        for item in inventory[:5]:  # Show first 5 items
            report += f"- {item['name']}: {item['current_quantity']} units\n"
        
        return report
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Medical Inventory Checker") as interface:
            gr.Markdown("# 🏥 Medical Inventory Checker")
            gr.Markdown("Upload before/after images of medical cases to detect missing or changed items")
            
            with gr.Row():
                with gr.Column():
                    before_input = gr.Image(
                        label="Before Image",
                        type="pil",
                        height=400
                    )
                    
                with gr.Column():
                    after_input = gr.Image(
                        label="After Image", 
                        type="pil",
                        height=400
                    )
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=self.available_models,
                    value=self.available_models[0],
                    label="Select AI Model"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.1,
                    label="Detection Threshold"
                )
            
            process_btn = gr.Button("🔍 Analyze Inventory", variant="primary")
            
            with gr.Row():
                output_image = gr.Image(
                    label="Difference Visualization",
                    type="pil"
                )
                
                output_report = gr.Markdown(label="Analysis Report")
            
            # Example images
            gr.Examples(
                examples=[
                    ["examples/before1.jpg", "examples/after1.jpg"],
                    ["examples/before2.jpg", "examples/after2.jpg"]
                ],
                inputs=[before_input, after_input],
                label="Example Cases"
            )
            
            # Connect the processing function
            process_btn.click(
                fn=self.process_images,
                inputs=[before_input, after_input, model_selector, threshold_slider],
                outputs=[output_image, output_report]
            )
            
        return interface

def create_interface():
    """Create Gradio interface - module level function for imports"""
    app = MedicalInventoryApp()
    return app.create_interface()

def main():
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()