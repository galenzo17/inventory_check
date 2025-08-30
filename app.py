#!/usr/bin/env python3
"""
Medical Inventory Checker - Hugging Face Spaces App
Main entry point for the Gradio application
"""

import gradio as gr
import numpy as np
from PIL import Image
import json
import os
import spaces

# Direct imports from root level (flat layout)
from inventory_checker import InventoryChecker
from visualization import create_difference_visualization
from inventory_db import InventoryDatabase

# Internationalization support - Custom implementation
class I18n:
    def __init__(self, translations):
        self.translations = translations
        self.current_lang = 'en'  # Default to English
    
    def set_language(self, lang):
        self.current_lang = lang if lang in self.translations else 'en'
    
    def __call__(self, key):
        return self.translations.get(self.current_lang, {}).get(key, 
                                     self.translations['en'].get(key, key))

i18n = I18n({
    'en': {
        "title": "🏥 Medical Inventory Checker",
        "subtitle": "Upload before/after images of medical cases to detect missing or changed items",
        "before_image": "Before Image",
        "after_image": "After Image",
        "select_model": "Select AI Model",
        "threshold": "Detection Threshold",
        "check_inventory": "🔍 Check Inventory",
        "analysis_results": "Analysis Results",
        "inventory_report": "Inventory Report",
        "example_cases": "Example Cases",
        "upload_both": "Please upload both before and after images",
        "error_processing": "Error processing images",
        # Report translations
        "report_title": "## Inventory Analysis Report\n\n",
        "dino_enhanced": "🔬 **Enhanced Analysis (DINO)**\n",
        "overall_similarity": "- Overall Similarity: ",
        "change_detected": "- Change Detected: ",
        "change_magnitude": "- Change Magnitude: ",
        "yes": "Yes",
        "no": "No",
        "no_differences": "✅ No significant differences detected\n",
        "found_differences": "⚠️ Found {} differences:\n\n",
        "item": "   - Item: ",
        "location": "   - Location: ",
        "confidence": "   - Confidence: ",
        "unknown": "Unknown",
        "na": "N/A",
        "expected_inventory": "\n## Expected Inventory:\n",
        "units": " units",
    },
    'es': {
        "title": "🏥 Verificador de Inventario Médico",
        "subtitle": "Sube imágenes de antes/después de cajas médicas para detectar elementos faltantes o cambiados",
        "before_image": "Imagen Anterior",
        "after_image": "Imagen Posterior",
        "select_model": "Seleccionar Modelo de IA",
        "threshold": "Umbral de Detección",
        "check_inventory": "🔍 Verificar Inventario",
        "analysis_results": "Resultados del Análisis",
        "inventory_report": "Reporte de Inventario",
        "example_cases": "Casos de Ejemplo",
        "upload_both": "Por favor sube ambas imágenes: antes y después",
        "error_processing": "Error al procesar las imágenes",
        # Report translations
        "report_title": "## Reporte de Análisis de Inventario\n\n",
        "dino_enhanced": "🔬 **Análisis Mejorado (DINO)**\n",
        "overall_similarity": "- Similitud General: ",
        "change_detected": "- Cambio Detectado: ",
        "change_magnitude": "- Magnitud del Cambio: ",
        "yes": "Sí",
        "no": "No",
        "no_differences": "✅ No se detectaron diferencias significativas\n",
        "found_differences": "⚠️ Se encontraron {} diferencias:\n\n",
        "item": "   - Elemento: ",
        "location": "   - Ubicación: ",
        "confidence": "   - Confianza: ",
        "unknown": "Desconocido",
        "na": "N/D",
        "expected_inventory": "\n## Inventario Esperado:\n",
        "units": " unidades",
    }
})

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
            return None, i18n("upload_both")
        
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
            return None, f"{i18n('error_processing')}: {str(e)}"
    
    def _generate_report(self, results):
        """Generate a human-readable report of inventory differences"""
        
        report = i18n("report_title")
        
        # Add DINO analysis if available
        if results.get('analysis', {}).get('enhanced_with_dino'):
            dino_sim = results['analysis']['dino_similarity']
            dino_change = results['analysis']['dino_change_detected']
            change_mag = results['analysis']['dino_change_magnitude']
            
            report += i18n("dino_enhanced")
            report += f"{i18n('overall_similarity')}{dino_sim:.3f}\n"
            report += f"{i18n('change_detected')}{i18n('yes') if dino_change else i18n('no')}\n"
            report += f"{i18n('change_magnitude')}{change_mag:.3f}\n\n"
        
        if not results['differences']:
            report += i18n("no_differences")
        else:
            report += i18n("found_differences").format(len(results['differences']))
            
            for i, diff in enumerate(results['differences'], 1):
                report += f"**{i}. {diff['type']}**\n"
                report += f"{i18n('item')}{diff.get('item', i18n('unknown'))}\n"
                report += f"{i18n('location')}{diff.get('location', i18n('na'))}\n"
                report += f"{i18n('confidence')}{diff.get('confidence', 0):.2%}\n\n"
        
        # Add inventory summary from database
        report += i18n("expected_inventory")
        inventory = self.db.get_all_items()
        for item in inventory[:5]:  # Show first 5 items
            report += f"- {item['name']}: {item['current_quantity']}{i18n('units')}\n"
        
        return report
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title=i18n("title")) as interface:
            # Language selector at the top
            with gr.Row():
                language_selector = gr.Radio(
                    choices=["English", "Español"],
                    value="English",
                    label="Language / Idioma",
                    scale=1
                )
            
            # Dynamic title and subtitle
            title_display = gr.Markdown(f"# {i18n('title')}")
            subtitle_display = gr.Markdown(i18n("subtitle"))
            
            with gr.Row():
                with gr.Column():
                    before_input = gr.Image(
                        label=i18n("before_image"),
                        type="pil",
                        height=400
                    )
                    
                with gr.Column():
                    after_input = gr.Image(
                        label=i18n("after_image"), 
                        type="pil",
                        height=400
                    )
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=self.available_models,
                    value=self.available_models[0],
                    label=i18n("select_model")
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.1,
                    label=i18n("threshold")
                )
            
            process_btn = gr.Button(i18n("check_inventory"), variant="primary")
            
            with gr.Row():
                with gr.Column():
                    output_image = gr.Image(
                        label=i18n("analysis_results"),
                        type="pil"
                    )
                
                with gr.Column():
                    output_report = gr.Markdown(
                        label=i18n("inventory_report")
                    )
            
            gr.Examples(
                examples=[],
                inputs=[before_input, after_input],
                label=i18n("example_cases")
            )
            
            # Language change handler
            def update_language(lang):
                i18n.set_language('es' if lang == "Español" else 'en')
                return (
                    f"# {i18n('title')}",
                    i18n("subtitle"),
                    gr.update(label=i18n("before_image")),
                    gr.update(label=i18n("after_image")),
                    gr.update(label=i18n("select_model")),
                    gr.update(label=i18n("threshold")),
                    gr.update(value=i18n("check_inventory")),
                    gr.update(label=i18n("analysis_results")),
                    gr.update(label=i18n("inventory_report"))
                )
            
            language_selector.change(
                fn=update_language,
                inputs=[language_selector],
                outputs=[
                    title_display, subtitle_display,
                    before_input, after_input,
                    model_selector, threshold_slider,
                    process_btn,
                    output_image, output_report
                ]
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
    interface = create_interface()
    interface.launch()