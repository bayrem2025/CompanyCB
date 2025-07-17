# image_analyzer.py

import base64
import io
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import cv2
import numpy as np
from typing import Dict, Any


class ImageAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=0 if torch.cuda.is_available() else -1)

    def encode_image_to_base64(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def decode_base64_to_image(self, base64_string: str) -> Image.Image:
        data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        image = image.convert("RGB")  # Assure RGB mode
        img_inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        output = self.blip_model.generate(**img_inputs, max_length=100)
        description = self.blip_processor.decode(output[0], skip_special_tokens=True)

        classes = self.classifier(image)[:5]
        tech = {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown')
        }
        color = self._analyze_colors(image)

        return {
            'description': description,
            'classifications': classes,
            'technical_info': tech,
            'color_analysis': color
        }

    def _analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image.resize((150, 150))).reshape(-1, 3).astype(np.float32)
        _, labels, centers = cv2.kmeans(arr, 5, None, 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            10, cv2.KMEANS_RANDOM_CENTERS)
        colors = [[int(c) for c in center] for center in centers]
        return {
            'dominant_colors': colors,
            'color_count': len(colors)
        }

    def generate_summary(self, result: Dict[str, Any]) -> str:
        parts = [
            f"Description: {result['description']}",
            f"Catégorie principale: {result['classifications'][0]['label']} ({result['classifications'][0]['score']:.2%})",
            f"Dimensions: {result['technical_info']['size'][0]}x{result['technical_info']['size'][1]} px",
            f"Couleurs dominantes: {result['color_analysis']['color_count']} principales"
        ]
        return " | ".join(parts)

    def process_image_complete(self, inp) -> Dict[str, Any]:
        try:
            if isinstance(inp, str) and (inp.startswith('data:') or len(inp) > 100):
                b64 = inp.split(',')[1] if inp.startswith('data:') else inp
                image = self.decode_base64_to_image(b64)
            else:
                image = Image.open(inp).convert("RGB")
                b64 = self.encode_image_to_base64(inp)

            analysis = self.analyze_image(image)
            summary = self.generate_summary(analysis)

            return {
                'base64': b64,
                'analysis': analysis,
                'summary': summary,
                'status': 'success'
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }