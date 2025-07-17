import os
import logging
from typing import List, Dict, Any, Optional
import uuid

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import Config # Assurez-vous que Config est correctement défini et importé

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
                logger.error(f"ERREUR CRITIQUE: Impossible de générer des embeddings avec Google Gemini. Vérifiez votre GOOGLE_API_KEY et les quotas. Erreur: {e}", exc_info=True)
                raise 

            # Utilisation de Chroma avec le répertoire de persistance
            self.vectorstore = Chroma(
                persist_directory=Config.CHROMA_PATH,
                embedding_function=self.embedding_function
            )
            logger.info(f"Vectorstore ChromaDB initialisé à: {Config.CHROMA_PATH}")

            # Initialisation du découpeur de texte pour le chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            logger.info(f"Text splitter initialisé (chunk_size={Config.CHUNK_SIZE}, chunk_overlap={Config.CHUNK_OVERLAP})")

        except Exception as e:
            logger.error(f"Erreur critique lors de l'initialisation de MultimodalRAGSystem: {e}", exc_info=True)
            raise

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        if not content or not content.strip():
            logger.warning("Tentative d'ajouter un document vide au système RAG.")
            return []

        doc_metadata = metadata if metadata is not None else {}
        if 'source' not in doc_metadata:
            doc_metadata['source'] = 'unknown_source'
        if 'uuid' not in doc_metadata: 
            doc_metadata['uuid'] = str(uuid.uuid4())

        try:
            # Crée un objet Document LangChain
            doc = Document(page_content=content, metadata=doc_metadata)
            
            # Découpe le document en chunks
            chunks = self.text_splitter.split_documents([doc])
            
            logger.debug(f"Processing document: {doc_metadata.get('source', 'N/A')} with {len(chunks)} chunks. First 100 chars: {content[:100]}...")
            if not chunks:
                logger.warning(f"Aucun chunk généré pour le document '{doc_metadata.get('source', 'N/A')}'. Le document était peut-être vide après extraction.")
                return []

            # Ajoute les chunks au vectorstore Chroma
            ids = self.vectorstore.add_documents(chunks)
            if not ids:
                logger.error(f"ChromaDB n'a retourné aucun ID pour le document '{doc_metadata.get('source', 'N/A')}'. L'indexation a échoué.")
                return []

            logger.info(f"Ajouté {len(ids)} chunks du document '{doc_metadata.get('source', 'N/A')}' à l'index. IDs: {ids[:3]}...")
            self.vectorstore.persist() # Sauvegarde les changements
            return ids
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du document au système RAG (source: {doc_metadata.get('source', 'N/A')}): {e}", exc_info=True)
            return []

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Performing similarity search for query: '{query}' with n_results={n_results}.")
            
            # Utilise similarity_search_with_score pour obtenir la pertinence
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_results) 
            
            if not results_with_scores:
                logger.info(f"No similarity search results found in ChromaDB for query: '{query}'")
            else:
                logger.info(f"Found {len(results_with_scores)} search results. Details:")
                for i, (doc, score) in enumerate(results_with_scores):
                    logger.info(f"  Result {i+1}: Score={score:.4f}, Source='{doc.metadata.get('source', 'N/A')}', Type='{doc.metadata.get('type', 'N/A')}', UUID='{doc.metadata.get('uuid', 'N/A')}', Content='{doc.page_content[:200]}...'")

            formatted_results = []
            for doc, score in results_with_scores:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "id": doc.metadata.get('uuid') # Assurez-vous que l'ID est dans les métadonnées
                })
            logger.info(f"Search for '{query}' completed. Retrieved {len(formatted_results)} formatted results.")
            return formatted_results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche dans le système RAG pour la requête '{query}': {e}", exc_info=True)
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        # Cette méthode est moins utile avec le chunking, car un ID de document original
        # peut correspondre à plusieurs chunks. Elle récupérerait un chunk spécifique.
        logger.warning(f"La récupération directe par ID '{doc_id}' dans ChromaDB peut ne pas retourner le document complet original. Cela récupérera un chunk spécifique.")
        try:
            # ChromaDB's .get() method often expects a list of IDs and returns structured data
            results = self.vectorstore.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
            if results and results['documents']:
                return {"content": results['documents'][0], "metadata": results['metadatas'][0]}
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document par ID '{doc_id}': {e}", exc_info=True)
            return None

    def delete_documents(self, doc_ids: List[str]):
        try:
            if doc_ids:
                # La méthode delete de Chroma peut prendre des IDs directement
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
            
            # Re-initialise le vectorstore après avoir vidé son répertoire
            self.vectorstore = Chroma(
                persist_directory=Config.CHROMA_PATH,
                embedding_function=self.embedding_function
            )
            logger.info("Vectorstore ChromaDB vidé et réinitialisé avec succès.")
            
        except Exception as e:
            logger.error(f"Erreur lors du vidage du vectorstore: {e}", exc_info=True)