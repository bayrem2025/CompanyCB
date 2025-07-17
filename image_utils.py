import base64
import logging
import os
from typing import Optional # <--- Ensure this is present

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> Optional[str]:
    """
    Encodes an image file to a base64 string.
    Returns None if the file does not exist or an error occurs.
    """
    if not os.path.exists(image_path): # <--- Added check
        logger.error(f"Image non trouvée: {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8') # <--- Explicit variable
            return encoded_string
    except Exception as e:
        logger.error(f"Erreur lors de l'encodage de l'image {image_path}: {e}")
        return None