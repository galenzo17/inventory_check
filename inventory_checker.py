import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForZeroShotObjectDetection
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from dino_extractor import DinoFeatureExtractor

class InventoryChecker:
    def __init__(self, model_name: str = "microsoft/kosmos-2-patch14-224", use_dino: bool = True):
        """Initialize the inventory checker with a specific model"""
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_dino = use_dino
        
        # Initialize DINO feature extractor if enabled
        if self.use_dino:
            try:
                self.dino_extractor = DinoFeatureExtractor(device=str(self.device))
            except Exception as e:
                print(f"Warning: Could not initialize DINO extractor: {e}")
                self.dino_extractor = None
                self.use_dino = False
        
        # Load model based on type with error handling
        try:
            if "kosmos" in model_name.lower():
                self._load_kosmos_model()
            elif "blip" in model_name.lower():
                self._load_blip_model()
            elif "qwen" in model_name.lower():
                self._load_qwen_model()
            elif "llama" in model_name.lower():
                self._load_llama_model()
            elif "florence" in model_name.lower():
                self._load_florence_model()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to Kosmos-2...")
            self.model_name = "microsoft/kosmos-2-patch14-224"
            self._load_kosmos_model()
    
    def _load_qwen_model(self):
        """Load Qwen2.5-VL model"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        
        # Use more conservative settings
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load with conservative settings to avoid state_dict issues
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cpu" if not torch.cuda.is_available() else "auto"
        )
        
    def _load_llama_model(self):
        """Load Llama 3.2 Vision model"""
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        import torch
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True,
            device_map="cpu" if not torch.cuda.is_available() else "auto"
        )
        
    def _load_kosmos_model(self):
        """Load Kosmos-2 model"""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        print(f"Loading Kosmos-2 model: {self.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        
        print("Kosmos-2 model loaded successfully")
        
    def _load_florence_model(self):
        """Load Microsoft Florence-2 model"""
        from transformers import AutoModelForCausalLM
        
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
    def _load_owlvit_model(self):
        """Load OWL-ViT model for open-vocabulary detection"""
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
        
    def _load_blip_model(self):
        """Load BLIP-2 model"""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
    
    def compare_inventory(self, before_image: Image.Image, after_image: Image.Image, 
                         threshold: float = 0.5) -> Dict:
        """Compare two images and detect inventory differences"""
        
        # Enhanced comparison with DINO features if available
        dino_results = None
        if self.use_dino and self.dino_extractor:
            try:
                dino_results = self.dino_extractor.detect_changes(
                    before_image, after_image, threshold=0.8
                )
            except Exception as e:
                print(f"Warning: DINO analysis failed: {e}")
        
        # Detect objects in both images using the main model
        before_objects = self._detect_objects(before_image)
        after_objects = self._detect_objects(after_image)
        
        # Find differences
        differences = self._find_differences(before_objects, after_objects, threshold)
        
        # Generate detailed analysis
        analysis = self._analyze_differences(differences, before_image, after_image)
        
        # Enhance analysis with DINO results if available
        if dino_results:
            analysis['dino_similarity'] = dino_results['similarity']
            analysis['dino_change_detected'] = dino_results['has_changed']
            analysis['dino_change_magnitude'] = dino_results['change_magnitude']
            analysis['enhanced_with_dino'] = True
        else:
            analysis['enhanced_with_dino'] = False
        
        return {
            'before_objects': before_objects,
            'after_objects': after_objects,
            'differences': differences,
            'analysis': analysis,
            'dino_results': dino_results
        }
    
    def _detect_objects(self, image: Image.Image) -> List[Dict]:
        """Detect objects in a single image"""
        
        if "qwen" in self.model_name.lower():
            return self._detect_qwen(image)
        elif "llama" in self.model_name.lower():
            return self._detect_llama(image)
        elif "kosmos" in self.model_name.lower():
            return self._detect_kosmos(image)
        elif "florence" in self.model_name.lower():
            return self._detect_florence(image)
        elif "owlvit" in self.model_name.lower():
            return self._detect_owlvit(image)
        elif "blip" in self.model_name.lower():
            return self._detect_blip(image)
        
    def _detect_qwen(self, image: Image.Image) -> List[Dict]:
        """Use Qwen2.5-VL for object detection"""
        
        # Detailed prompt for medical inventory analysis
        prompt = """Analyze this medical equipment case image and identify all visible items. For each item, provide:
1. Item name (be specific: type of screw, tool name, etc.)
2. Approximate location/position
3. Count if multiple items

Focus on medical equipment like:
- Surgical screws (cortical, cancellous, locking)
- Medical instruments (forceps, drivers, gauges)
- Drill bits and guides
- Plates and implants
- Any tools or components

Provide a detailed list of everything visible."""

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.processor.process(text=text, images=[image], videos=None, return_tensors="pt")
        
        inputs = {
            "input_ids": image_inputs["input_ids"].to(self.device),
            "pixel_values": image_inputs["pixel_values"].to(self.device),
            "attention_mask": image_inputs["attention_mask"].to(self.device),
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Parse the response to extract objects
        return self._parse_text_to_objects(response)
        
    def _detect_llama(self, image: Image.Image) -> List[Dict]:
        """Use Llama 3.2 Vision for object detection"""
        
        prompt = """<|image|>Carefully examine this medical equipment case and list all visible items. For each item, specify:
- Exact item name (e.g., "3.5mm cortical screw", "hex screwdriver", "depth gauge")  
- Approximate position in the image
- Quantity if multiple

Focus on medical/surgical equipment like screws, instruments, drill bits, plates, and tools."""

        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
        
        response = self.processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Parse the response to extract objects  
        return self._parse_text_to_objects(response)
        
    def _detect_kosmos(self, image: Image.Image) -> List[Dict]:
        """Use Kosmos-2 for object detection"""
        
        prompt = "<grounding>Describe all medical equipment and tools visible in this surgical case, including their locations."
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=512,
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse Kosmos-2 output which includes bounding boxes
        return self._parse_kosmos_output(generated_text, image)
        
    def _parse_text_to_objects(self, response: str) -> List[Dict]:
        """Parse text response to extract objects"""
        
        objects = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('##'):
                continue
                
            # Look for item descriptions
            if any(keyword in line.lower() for keyword in ['screw', 'drill', 'tool', 'instrument', 'gauge', 'driver', 'plate', 'forceps']):
                # Extract item name (simplified parsing)
                item_name = line
                if ':' in line:
                    item_name = line.split(':')[0].strip()
                if '-' in item_name:
                    item_name = item_name.split('-', 1)[1].strip()
                    
                objects.append({
                    'label': item_name,
                    'bbox': None,  # These models don't provide precise bboxes
                    'confidence': 0.85
                })
        
        return objects
        
    def _parse_kosmos_output(self, text: str, image: Image.Image) -> List[Dict]:
        """Parse Kosmos-2 output with grounding boxes"""
        
        objects = []
        # Kosmos-2 uses special tokens for grounding
        import re
        
        # Extract grounded phrases and their bounding boxes
        pattern = r'<phrase>([^<]+)</phrase><object>([^<]+)</object>'
        matches = re.findall(pattern, text)
        
        for phrase, bbox_str in matches:
            try:
                # Parse bounding box coordinates
                coords = [float(x) for x in bbox_str.split(',')]
                if len(coords) == 4:
                    # Convert normalized coordinates to pixel coordinates
                    w, h = image.size
                    x1, y1, x2, y2 = coords
                    bbox = [x1 * w, y1 * h, x2 * w, y2 * h]
                    
                    objects.append({
                        'label': phrase.strip(),
                        'bbox': bbox,
                        'confidence': 0.9
                    })
            except:
                # If parsing fails, add without bbox
                objects.append({
                    'label': phrase.strip(),
                    'bbox': None,
                    'confidence': 0.8
                })
        
        return objects
        
    def _detect_florence(self, image: Image.Image) -> List[Dict]:
        """Use Florence-2 for object detection"""
        
        task_prompt = "<DENSE_REGION_CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_results = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        # Convert to standardized format
        objects = []
        if 'labels' in parsed_results:
            for i, label in enumerate(parsed_results['labels']):
                if i < len(parsed_results.get('bboxes', [])):
                    bbox = parsed_results['bboxes'][i]
                    objects.append({
                        'label': label,
                        'bbox': bbox,
                        'confidence': 0.9  # Florence doesn't provide confidence
                    })
        
        return objects
    
    def _detect_owlvit(self, image: Image.Image) -> List[Dict]:
        """Use OWL-ViT for object detection"""
        
        # Medical inventory related queries
        texts = [
            "medical screw", "surgical tool", "medical instrument",
            "forceps", "scalpel", "drill bit", "implant",
            "medical device", "surgical equipment"
        ]
        
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=0.1
        )[0]
        
        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            objects.append({
                'label': texts[label],
                'bbox': box.tolist(),
                'confidence': score.item()
            })
        
        return objects
    
    def _detect_blip(self, image: Image.Image) -> List[Dict]:
        """Use BLIP-2 for image analysis"""
        
        # Use VQA for inventory analysis
        question = "What medical tools and equipment are visible in this image?"
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=100)
        
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Parse answer to extract objects (simplified)
        objects = []
        items = answer.split(',')
        for i, item in enumerate(items):
            objects.append({
                'label': item.strip(),
                'bbox': None,  # BLIP doesn't provide bboxes
                'confidence': 0.8
            })
        
        return objects
    
    def _find_differences(self, before: List[Dict], after: List[Dict], 
                         threshold: float) -> List[Dict]:
        """Find differences between before and after object lists"""
        
        differences = []
        
        # Track matched objects
        matched_after = set()
        
        # Find missing objects (in before but not in after)
        for b_obj in before:
            found = False
            for i, a_obj in enumerate(after):
                if i not in matched_after:
                    if self._objects_match(b_obj, a_obj, threshold):
                        found = True
                        matched_after.add(i)
                        break
            
            if not found:
                differences.append({
                    'type': 'missing',
                    'item': b_obj['label'],
                    'location': b_obj.get('bbox'),
                    'confidence': b_obj['confidence']
                })
        
        # Find added objects (in after but not in before)
        for i, a_obj in enumerate(after):
            if i not in matched_after:
                differences.append({
                    'type': 'added',
                    'item': a_obj['label'],
                    'location': a_obj.get('bbox'),
                    'confidence': a_obj['confidence']
                })
        
        return differences
    
    def _objects_match(self, obj1: Dict, obj2: Dict, threshold: float) -> bool:
        """Check if two objects match based on label and location"""
        
        # Check label similarity
        if obj1['label'].lower() == obj2['label'].lower():
            # If both have bboxes, check spatial overlap
            if obj1.get('bbox') and obj2.get('bbox'):
                iou = self._calculate_iou(obj1['bbox'], obj2['bbox'])
                return iou > threshold
            return True
        
        return False
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        
        # Convert to numpy arrays
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_differences(self, differences: List[Dict], 
                           before_image: Image.Image, 
                           after_image: Image.Image) -> Dict:
        """Generate detailed analysis of differences"""
        
        analysis = {
            'total_differences': len(differences),
            'missing_items': sum(1 for d in differences if d['type'] == 'missing'),
            'added_items': sum(1 for d in differences if d['type'] == 'added'),
            'critical_items': [],
            'recommendations': []
        }
        
        # Identify critical items (surgical tools, implants)
        critical_keywords = ['screw', 'implant', 'scalpel', 'forceps', 'drill']
        for diff in differences:
            if any(keyword in diff['item'].lower() for keyword in critical_keywords):
                analysis['critical_items'].append(diff['item'])
        
        # Generate recommendations
        if analysis['missing_items'] > 0:
            analysis['recommendations'].append(
                f"Restock {analysis['missing_items']} missing items"
            )
        
        if analysis['critical_items']:
            analysis['recommendations'].append(
                f"Priority: Replace critical items: {', '.join(analysis['critical_items'])}"
            )
        
        return analysis