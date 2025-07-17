import google.generativeai as genai
from config import Config
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=Config.GOOGLE_API_KEY)

def image_summarize(base64_image: str) -> str:
    try:
        model = genai.GenerativeModel(Config.GEMINI_VISION_MODEL)
        
        # Prompt dynamique pour l'analyse d'image
        # Il demande à Gemini d'être spécifique sur les drapeaux ou les diagrammes UML
        vision_prompt = """Décris cette image en détail. 
        
        Si l'image est un **drapeau**, décris précisément ses couleurs, leur ordre, et son orientation (verticale/horizontale). N'identifie pas le pays directement, mais donne des informations très spécifiques sur le motif et les couleurs.

        Si l'image est un **diagramme UML**, identifie toutes les classes, leurs attributs et surtout leurs **opérations/méthodes** avec leur signature (nom, paramètres). Liste chaque classe avec ses méthodes. Si un texte est présent, retranscris-le.
        
        Si ce n'est ni un drapeau ni un diagramme UML, donne une description générale mais détaillée.
        """

        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": base64_image}, # Assumons jpeg par défaut, ou adaptez le mime_type
            vision_prompt
        ])
        
        # Vérifiez si la réponse est vide ou non pertinente
        if not response.text or response.text.strip().lower() in ["échec de l'analyse", "aucune information pertinente", "je ne peux pas analyser"]:
            logger.warning(f"Gemini Vision a retourné un résumé vide ou non pertinent. Prompt utilisé: '{vision_prompt[:100]}...'")
            return "Échec de l’analyse de l’image ou résumé vide."
            
        logger.info(f"Résumé d'image généré par Gemini: {response.text[:200]}...")
        return response.text
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à Gemini Vision (image_summarize): {e}", exc_info=True)
        return "Échec de l’analyse de l’image en raison d'une erreur technique."