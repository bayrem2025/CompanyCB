from utils.image_utils import encode_image
from config import Config
from vectordb import ChromaVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import logging
from universal_processor import load_and_process # New import

logger = logging.getLogger(__name__)

class MultimodalRAGSystem:
    def __init__(self):
        try:
            Config.create_directories()

            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model=Config.GOOGLE_GEMINI_EMBEDDING_MODEL,
                google_api_key=Config.GOOGLE_API_KEY
            )
            logger.info(f"Fonction d'embedding chargée: {Config.GOOGLE_GEMINI_EMBEDDING_MODEL} (via Google Gemini).")

            try:
                # Test de la fonction d'embedding
                test_embedding = self.embedding_function.embed_query("test query")
                if not test_embedding or len(test_embedding) == 0:
                    raise ValueError("Google Gemini a retourné un embedding vide pour la requête de test.")
                logger.info("Google Gemini Embeddings initialisé et testé avec succès.")
            except Exception as e:
                logger.error(f"ERREUR CRITIQUE: Impossible de générer des embeddings avec Google Gemini. Vérifiez votre GOOGLE_API_KEY et les quotas. Erreur: {e}")
                raise

            self.vectorstore = ChromaVectorStore(
                persist_directory=Config.CHROMA_PATH,
                embedding_function=self.embedding_function
            ).create_or_load()
            logger.info(f"Vectorstore ChromaDB chargé/créé à: {Config.CHROMA_PATH}.")

        except Exception as e:
            logger.exception(f"Échec de l'initialisation de MultimodalRAGSystem: {e}")
            raise

    def index_text_file(self, file_path: str):
        # Use load_and_process from universal_processor
        documents = load_and_process(file_path)
        if not documents:
            logger.warning(f"Fichier vide ou non parsable: {file_path}. Non indexé.")
            return None

        # Add metadata to each document
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["type"] = "text" # Or infer type from original loader if needed

        self.vectorstore.add_documents(documents)
        logger.info(f"Fichier texte indexé: {file_path}")
        return documents # Return the list of documents

    def index_image_file(self, image_path: str):
        encoded = encode_image(image_path)
        if not encoded:
            logger.error(f"Échec de l'encodage de l'image pour l'indexation: {image_path}")
            return None

        # Use Config.GEMINI_VISION_MODEL for consistency
        model = ChatGoogleGenerativeAI(model=Config.GEMINI_VISION_MODEL, google_api_key=Config.GOOGLE_API_KEY)
        prompt = f"Describe this image in detail (format: plain text): <image>{encoded}</image>"

        try:
            response = model.invoke(prompt)
            text = response.content.strip()
            if not text:
                logger.warning(f"La description de l'image est vide pour: {image_path}. Non indexé.")
                return None
            doc = Document(page_content=text, metadata={"source": image_path, "type": "image"})
            self.vectorstore.add_documents([doc])
            logger.info(f"Fichier image indexé: {image_path}")
            return doc
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse et de l'indexation de l'image '{image_path}': {e}", exc_info=True)
            return None

    def search(self, query: str, n_results: int = 3):
        logger.info(f"Recherche de documents similaires pour la requête: '{query}'")
        try:
            results = self.vectorstore.similarity_search(query, k=n_results)
            logger.info(f"Trouvé {len(results)} résultats pour la requête.")
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de similarité: {e}", exc_info=True)
            return []

    def get_document_by_id(self, doc_id: str):
        try:
            logger.warning(f"La récupération de document par ID '{doc_id}' n'est pas directement supportée par l'API simplifiée. Récupération via recherche non implémentée.")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document par ID '{doc_id}': {e}", exc_info=True)
            return None

    def delete_documents(self, doc_ids: list[str]):
        try:
            if doc_ids:
                self.vectorstore.delete(ids=doc_ids)
                logger.info(f"Supprimé les documents (chunks) avec les IDs: {doc_ids[:5]}...")
                self.vectorstore.persist()
            else:
                logger.info("Aucun ID de document fourni pour la suppression.")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des documents: {e}", exc_info=True)

    def clear_vectorstore(self):
        try:
            if os.path.exists(Config.CHROMA_PATH):
                import shutil
                shutil.rmtree(Config.CHROMA_PATH)
                logger.info(f"Répertoire ChromaDB vidé: {Config.CHROMA_PATH}.")

            self.vectorstore = ChromaVectorStore(
                persist_directory=Config.CHROMA_PATH,
                embedding_function=self.embedding_function
            ).create_or_load()
            logger.info("Vectorstore ChromaDB vidé et réinitialisé avec succès.")

        except Exception as e:
            logger.error(f"Erreur lors du vidage du vectorstore ChromaDB: {e}", exc_info=True)

def image_summarize(img_base64: str, prompt: str = None) -> str:
    """Summarizes an image using the Gemini Vision model."""
    if not prompt:
        prompt = """Décris cette image en détail en français. Inclus :
        - Objets principaux et leur disposition
        - Personnes (nombre, apparence, actions)
        - Contexte et ambiance
        - Couleurs dominantes
        - Texte visible"""

    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_VISION_MODEL,
        temperature=0,
        google_api_key=Config.GOOGLE_API_KEY
    )

    msg = llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
        ])
    ])

    return msg.content