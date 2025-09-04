#!/usr/bin/env python3
"""
Medical Item Segmentation App - Nueva vista de segmentación DINO
Permite subir una imagen y detectar elementos médicos con contornos
"""

import gradio as gr
import numpy as np
from PIL import Image
import json
import os
import spaces
import traceback

# Import from src structure
try:
    from src.models.dinov2_segmenter import DINOv2MedicalSegmenter
except ImportError:
    # Fallback if src structure not available
    import sys
    sys.path.append('src/models')
    from dinov2_segmenter import DINOv2MedicalSegmenter

# Internationalization support
class I18n:
    def __init__(self, translations):
        self.translations = translations
        self.current_lang = 'en'
    
    def set_language(self, lang):
        self.current_lang = lang if lang in self.translations else 'en'
    
    def __call__(self, key):
        return self.translations.get(self.current_lang, {}).get(key, 
                                     self.translations['en'].get(key, key))

i18n = I18n({
    'en': {
        "title": "🔬 Medical Item Segmentation",
        "subtitle": "Upload an image to detect and segment medical items using advanced DINOv2",
        "upload_image": "Upload Medical Image",
        "model_select": "DINOv2 Model Size",
        "threshold": "Detection Threshold",
        "min_size": "Minimum Object Size",
        "segment_button": "🔍 Segment Medical Items",
        "results_image": "Segmentation Results",
        "results_info": "Detection Information",
        "processing_error": "Error processing image",
        "no_image": "Please upload an image first",
        "detected_items": "Detected Items: ",
        "total_area": "Total Detection Area: ",
        "avg_confidence": "Average Confidence: ",
        "pixels": " pixels²",
        "instructions": """
        ## Instructions:
        1. Upload a medical image containing items like syringes, bandages, pills, etc.
        2. Adjust detection threshold (lower = more sensitive)
        3. Set minimum object size to filter small noise
        4. Click 'Segment Medical Items' to analyze
        
        **Features:**
        - Advanced DINOv2 attention-based detection
        - Real-time contour visualization  
        - No counting - pure detection and segmentation
        - Optimized for medical inventory items
        """,
        "example_cases": "Example Images"
    },
    'es': {
        "title": "🔬 Segmentación de Elementos Médicos",
        "subtitle": "Sube una imagen para detectar y segmentar elementos médicos usando DINOv2 avanzado",
        "upload_image": "Subir Imagen Médica",
        "model_select": "Tamaño del Modelo DINOv2",
        "threshold": "Umbral de Detección",
        "min_size": "Tamaño Mínimo de Objeto",
        "segment_button": "🔍 Segmentar Elementos Médicos",
        "results_image": "Resultados de Segmentación",
        "results_info": "Información de Detección",
        "processing_error": "Error al procesar la imagen",
        "no_image": "Por favor sube una imagen primero",
        "detected_items": "Elementos Detectados: ",
        "total_area": "Área Total de Detección: ",
        "avg_confidence": "Confianza Promedio: ",
        "pixels": " píxeles²",
        "instructions": """
        ## Instrucciones:
        1. Sube una imagen médica con elementos como jeringas, vendas, píldoras, etc.
        2. Ajusta el umbral de detección (más bajo = más sensible)
        3. Define el tamaño mínimo del objeto para filtrar ruido
        4. Haz clic en 'Segmentar Elementos Médicos' para analizar
        
        **Características:**
        - Detección avanzada basada en atención DINOv2
        - Visualización de contornos en tiempo real
        - Sin conteo - detección y segmentación pura
        - Optimizado para elementos de inventario médico
        """,
        "example_cases": "Imágenes de Ejemplo"
    }
})

class MedicalSegmentationApp:
    def __init__(self):
        self.segmenter = None
        self.available_models = ["vits14", "vitb14", "vitl14"]
        
    @spaces.GPU(duration=60)
    def segment_image(self, image, model_size, threshold, min_object_size):
        """Process image and return segmentation results"""
        
        if image is None:
            return None, i18n("no_image")
        
        try:
            # Initialize or update segmenter
            if self.segmenter is None or self.segmenter.model_size != model_size:
                self.segmenter = DINOv2MedicalSegmenter(model_size=model_size)
            
            # Perform segmentation
            segmentation_results = self.segmenter.segment_medical_items(
                image, 
                threshold=threshold,
                min_object_size=min_object_size
            )
            
            # Create visualization
            result_image = self.segmenter.visualize_segmentation(
                image, 
                segmentation_results,
                show_confidence=True
            )
            
            # Generate info report
            info_report = self._generate_info_report(segmentation_results)
            
            return result_image, info_report
            
        except Exception as e:
            error_msg = f"{i18n('processing_error')}: {str(e)}\n\nDetailed error:\n{traceback.format_exc()}"
            return None, error_msg
    
    def _generate_info_report(self, results):
        """Generate information report about detection results"""
        
        detected_objects = results['detected_objects']
        num_objects = results['num_objects']
        
        report = f"## {i18n('results_info')}\n\n"
        
        # Summary statistics
        report += f"**{i18n('detected_items')}** {num_objects}\n"
        
        if num_objects > 0:
            total_area = sum(obj['area'] for obj in detected_objects)
            avg_confidence = np.mean([obj['confidence'] for obj in detected_objects])
            
            report += f"**{i18n('total_area')}** {total_area:,}{i18n('pixels')}\n"
            report += f"**{i18n('avg_confidence')}** {avg_confidence:.3f}\n\n"
            
            # Individual object details
            report += "### Objetos Detectados / Detected Objects:\n\n"
            for i, obj in enumerate(detected_objects, 1):
                bbox = obj['bbox']
                report += f"**{i}.** Área: {obj['area']:,} px² | "
                report += f"Confianza: {obj['confidence']:.3f} | "
                report += f"Región: ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})\n"
        else:
            report += "\n❌ No se detectaron elementos médicos / No medical items detected\n"
            report += "💡 Prueba ajustar el umbral o el tamaño mínimo / Try adjusting threshold or minimum size"
        
        return report
    
    def create_interface(self):
        """Create Gradio interface for medical segmentation"""
        
        with gr.Blocks(title=i18n("title"), theme=gr.themes.Soft()) as interface:
            
            # Header with language selector
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML("""
                    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%); border-radius: 15px; margin-bottom: 20px;">
                        <h1 style="color: white; margin: 0; font-size: 2.2em;">🔬 Medical Item Segmentation</h1>
                        <p style="color: white; margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1em;">Advanced DINOv2-powered medical object detection</p>
                    </div>
                    """)
                with gr.Column(scale=1):
                    language_selector = gr.Radio(
                        choices=["English", "Español"],
                        value="English",
                        label="🌐 Language / Idioma"
                    )
            
            # Instructions
            instructions_display = gr.Markdown(i18n("instructions"))
            
            with gr.Row():
                # Left column - Input controls
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label=i18n("upload_image"),
                        type="pil",
                        height=400
                    )
                    
                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=self.available_models,
                            value="vits14",
                            label=i18n("model_select")
                        )
                    
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.6,
                        step=0.05,
                        label=i18n("threshold")
                    )
                    
                    min_size_slider = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=200,
                        step=50,
                        label=i18n("min_size")
                    )
                    
                    segment_btn = gr.Button(
                        i18n("segment_button"), 
                        variant="primary",
                        size="lg"
                    )
                
                # Right column - Results
                with gr.Column(scale=1):
                    result_image = gr.Image(
                        label=i18n("results_image"),
                        type="pil",
                        height=400
                    )
                    
                    result_info = gr.Markdown(
                        label=i18n("results_info"),
                        value="Sube una imagen y haz clic en 'Segmentar' / Upload image and click 'Segment'"
                    )
            
            # Example images section
            gr.Examples(
                examples=[],
                inputs=[image_input],
                label=i18n("example_cases")
            )
            
            # Language change handler
            def update_language(lang):
                i18n.set_language('es' if lang == "Español" else 'en')
                return (
                    i18n("instructions"),
                    gr.update(label=i18n("upload_image")),
                    gr.update(label=i18n("model_select")),
                    gr.update(label=i18n("threshold")),
                    gr.update(label=i18n("min_size")),
                    gr.update(value=i18n("segment_button")),
                    gr.update(label=i18n("results_image")),
                    "Sube una imagen y haz clic en 'Segmentar' / Upload image and click 'Segment'"
                )
            
            language_selector.change(
                fn=update_language,
                inputs=[language_selector],
                outputs=[
                    instructions_display,
                    image_input, model_selector,
                    threshold_slider, min_size_slider,
                    segment_btn, result_image, result_info
                ]
            )
            
            # Connect segmentation function
            segment_btn.click(
                fn=self.segment_image,
                inputs=[image_input, model_selector, threshold_slider, min_size_slider],
                outputs=[result_image, result_info]
            )
            
        return interface

def create_segmentation_interface():
    """Create segmentation interface - module level function"""
    app = MedicalSegmentationApp()
    return app.create_interface()

def main():
    interface = create_segmentation_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True
    )

if __name__ == "__main__":
    interface = create_segmentation_interface()
    interface.launch()