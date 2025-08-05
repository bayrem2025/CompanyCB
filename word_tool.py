from langchain.tools import tool
from utils.word_generator import generate_word_document
@tool("Word Document Generator")
def word_tool(data: str) -> str:
    """
    Génère un document Word à partir d'un sujet et de contenu texte.
    Format attendu :
    Sujet: <nom>
    
    Contenu:
    <texte formaté à inclure dans le document>
    """
    try:
        # Nettoyer et normaliser le texte d'entrée
        data = data.strip()
        
        # Vérifier les variantes possibles de l'en-tête
        if not (data.startswith("Sujet:") or "Contenu:" not in data):
            return "Erreur : format invalide. Le format doit être 'Sujet: <titre>\n\nContenu:\n<contenu>'"

        # Séparer le sujet et le contenu
        parts = data.split("Contenu:", 1)
        topic_part = parts[0].replace("Sujet:", "").strip()
        content_part = parts[1].strip()
        
        path = generate_word_document(topic_part, content_part)
        return f"Document généré avec succès : {path}"
    except Exception as e:
        return f"Erreur lors de la génération du document Word : {str(e)}"