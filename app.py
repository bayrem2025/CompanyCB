import os
import logging
from typing import Optional, List

import chainlit as cl
import torch
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS

from config import Config
from image_analyzer import ImageAnalyzer
from rag_system import MultimodalRAGSystem

# Import functions from the new file_handlers
from file_handlers import handle_attachments, set_global_instances as set_file_handlers_instances

# NEW: Import functions from the new query_handlers
from query_handlers import handle_text_query, set_rag_system_instance as set_query_handlers_rag_system_instance

# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
rag_system: Optional[MultimodalRAGSystem] = None
image_analyzer: Optional[ImageAnalyzer] = None

@cl.on_chat_start
async def init_rag_system():
    global rag_system, image_analyzer
    try:
        Config.create_directories()
        await cl.Message(content="🚀 Initialisation du système RAG multimodal...").send()

        rag_system = MultimodalRAGSystem()

        await cl.Message(content="🧠 Chargement des modèles d'analyse d'image (BLIP, ViT, OpenCV)...").send()
        image_analyzer = ImageAnalyzer()
        await cl.Message(content="✅ Modèles d'analyse d'image chargés.").send()

        # Pass the initialized instances to file_handlers
        set_file_handlers_instances(rag_system, image_analyzer)
        # NEW: Pass the initialized RAG system to query_handlers
        set_query_handlers_rag_system_instance(rag_system)


        await cl.Message(content="✅ Système prêt à traiter documents et images.").send()
    except Exception as e:
        await cl.Message(content=f"❌ Échec de l'initialisation du système: {str(e)}. Veuillez vérifier les configurations et assurez-vous que les modèles d'AI sont accessibles.").send()
        logger.exception("Failed to initialize RAG system.")
        raise

@cl.on_message
async def process_message(message: cl.Message):
    # global rag_system # Not directly used here, passed via set_rag_system_instance

    if not message.content and not getattr(message, 'elements', []):
        await cl.Message(content="❌ Veuillez poser une question ou envoyer un fichier à analyser.").send()
        return

    if hasattr(message, 'elements') and message.elements:
        await handle_attachments(message.elements) # Imported from file_handlers

    if message.content and message.content.strip():
        await handle_text_query(message.content) # Now imported from query_handlers

# The perform_web_search function is now in query_handlers.py and called from there.
# async def perform_web_search(query: str, llm):
#    ... (removed) ...

# The handle_text_query function is now in query_handlers.py
# async def handle_text_query(query: str):
#    ... (removed) ...

if __name__ == "__main__":
    cl.run()
