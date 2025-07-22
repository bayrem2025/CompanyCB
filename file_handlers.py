import os
import uuid
import logging
from typing import List, Optional

import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI # Only if used in process_image_file
from config import Config
from image_analyzer import ImageAnalyzer
from rag_system import MultimodalRAGSystem # Assuming rag_system is an instance of this
from utils.file_utils import (
    extract_text_from_txt,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_excel,
    extract_text_from_pptx
)

logger = logging.getLogger(__name__)

# Global instances (will be passed from app.py)
_rag_system: Optional[MultimodalRAGSystem] = None
_image_analyzer: Optional[ImageAnalyzer] = None

# Function to set global instances from app.py
def set_global_instances(rag_sys: MultimodalRAGSystem, img_analyzer: ImageAnalyzer):
    global _rag_system, _image_analyzer
    _rag_system = rag_sys
    _image_analyzer = img_analyzer
    logger.info("RAG system and Image Analyzer instances set in file_handlers.")

SUPPORTED_FILE_TYPES = {
    '.txt': 'Texte',
    '.docx': 'Word',
    '.pdf': 'PDF',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
    '.csv': 'CSV',
    '.pptx': 'PowerPoint',
    '.jpg': 'Image',
    '.jpeg': 'Image',
    '.png': 'Image',
    '.svg': 'Image',
    '.gif': 'Image'
}

async def process_image_file(element: cl.File):
    await cl.Message(content=f"🖼️ Analyse de l'image '{element.name}' en cours...").send()
    logger.info(f"Starting image processing for: {element.name}")
    try:
        if not _image_analyzer:
            raise ValueError("ImageAnalyzer not initialized in file_handlers.")
        if not _rag_system:
            raise ValueError("RAGSystem not initialized in file_handlers.")

        processed_data = _image_analyzer.process_image_complete(element.path)

        if processed_data['status'] == 'error':
            await cl.Message(content=f"❌ Échec de l'analyse de l'image '{element.name}': {processed_data['error']}").send()
            logger.warning(f"Image analysis failed for {element.name}: {processed_data['error']}.")
            return

        summary = processed_data['summary']
        full_analysis = processed_data['analysis']

        identified_country = "Non identifié"
        is_flag_classification = any("flag" in c['label'].lower() for c in full_analysis.get('classifications', [])[:2])

        if is_flag_classification:
            await cl.Message(content=f"🤔 Tentative d'identification du pays à partir de l'image...").send()
            try:
                llm_for_identification = ChatGoogleGenerativeAI(model=Config.GOOGLE_GEMINI_LLM_MODEL, temperature=0.1, google_api_key=Config.GOOGLE_API_KEY)

                identification_prompt = f"""Vous êtes un expert en drapeaux. En vous basant **uniquement** sur la description fournie de l'image, nommez le pays correspondant à ce drapeau.

                Description de l'image:
                {summary}

                Si la description correspond clairement à un drapeau, répondez uniquement avec le nom du pays.
                Si la description ne suffit pas pour identifier un pays avec certitude, répondez "Pays non identifiable avec les informations fournies."

                Exemples:
                - Description: "Description: a flag with three horizontal stripes of black, red, and yellow" -> Allemagne
                - Description: "Description: a vertical tricolor flag with green, white, and red stripes" -> Italie
                - Description: "Description: a red banner with a white circle and a red sun" -> Japon

                Quel est le pays correspondant à ce drapeau ?
                Réponse:"""

                llm_response = await llm_for_identification.ainvoke(identification_prompt)
                llm_identified_country = llm_response.content.strip()

                if "non identifiable" not in llm_identified_country.lower() and llm_identified_country:
                    identified_country = llm_identified_country
                    await cl.Message(content=f"✅ Pays identifié: **{identified_country}**.").send()
                else:
                    await cl.Message(content=f"ℹ️ Le pays n'a pas pu être identifié avec certitude à partir de la description du drapeau. ({llm_identified_country})").send()

            except Exception as llm_e:
                logger.error(f"Erreur lors de l'identification du pays par LLM: {llm_e}", exc_info=True)
                await cl.Message(content=f"❌ Erreur lors de l'identification du pays par l'IA: {str(llm_e)}").send()

        doc_content = (f"Résumé de l'image : {summary}\n"
                       f"Description détaillée: {full_analysis.get('description', 'N/A')}\n"
                       f"Classifications: {', '.join([c['label'] for c in full_analysis.get('classifications', [])])}\n"
                       f"Couleurs: {full_analysis.get('color_analysis', {}).get('dominant_colors', 'N/A')}\n"
                       f"Pays identifié (si applicable): {identified_country}")

        doc_ids = _rag_system.add_document(
            content=doc_content,
            metadata={
                "source": element.name,
                "type": "image_analysis",
                "original_summary": summary,
                "image_classifications": [c['label'] for c in full_analysis.get('classifications', [])],
                "image_description": full_analysis.get('description', ''),
                "image_dominant_colors": full_analysis.get('color_analysis', {}).get('dominant_colors', ''),
                "identified_country": identified_country,
                "uuid": str(uuid.uuid4())
            }
        )
        if not doc_ids:
            await cl.Message(content=f"⚠️ Le fichier '{element.name}' n'a pas été indexé car aucun contenu utile n'a été généré ou un problème est survenu lors de l'ajout à l'index.").send()
            logger.warning(f"No document IDs returned for {element.name}. Content might have been empty or indexing failed.")
            return

        await cl.Message(content=f"✅ Image '{element.name}' analysée et indexée (ID: `{doc_ids[0] if doc_ids else 'N/A'}`). Résumé: {summary[:100]}... Pays: {identified_country}").send()
        logger.info(f"Image '{element.name}' analyzed and indexed successfully. IDs: {doc_ids}")
    except Exception as e:
        await cl.Message(content=f"❌ Erreur traitement image '{element.name}': {str(e)}. Vérifiez votre `image_analyzer.py` et les dépendances.").send()
        logger.exception(f"Error processing image {element.name}.")
        return

async def process_standard_file(element: cl.File, file_ext: str, file_name: str):
    file_type = SUPPORTED_FILE_TYPES.get(file_ext, "Document")
    await cl.Message(content=f"📄 Lecture du fichier {file_type} '{file_name}' en cours...").send()
    logger.info(f"Starting standard file processing for: {file_name}")

    try:
        content = None
        if file_ext == '.txt':
            content = extract_text_from_txt(element.path)
        elif file_ext == '.docx':
            content = extract_text_from_docx(element.path)
        elif file_ext == '.pdf':
            content = extract_text_from_pdf(element.path)
        elif file_ext in ('.xlsx', '.xls', '.csv'):
            content = extract_text_from_excel(element.path)
        elif file_ext == '.pptx':
            content = extract_text_from_pptx(element.path)

        if not content:
            await cl.Message(content=f"❌ Le contenu du fichier '{file_name}' est vide ou illisible. Aucun texte n'a pu être extrait.").send()
            logger.warning(f"No content extracted from file {file_name}.")
            return
        if not _rag_system:
            raise ValueError("RAGSystem not initialized in file_handlers.")

        doc_ids = _rag_system.add_document(
            content=content,
            metadata={
                "source": file_name,
                "type": file_ext[1:],
                "uuid": str(uuid.uuid4())
            }
        )
        await cl.Message(content=f"✅ Fichier {file_type} '{file_name}' indexé (ID: `{doc_ids[0] if doc_ids else 'N/A'}`).").send()
        logger.info(f"Standard file '{file_name}' indexed successfully. IDs: {doc_ids}")
    except Exception as e:
        await cl.Message(content=f"❌ Erreur lors de l'extraction ou de l'indexation du fichier '{file_name}': {str(e)}. Le fichier est peut-être corrompu ou le format n'est pas entièrement pris en charge.").send()
        logger.exception(f"Error processing standard file {file_name}.")

async def handle_attachments(elements: List):
    logger.info(f"Handling {len(elements)} attachments.")
    for element in elements:
        file_path = getattr(element, 'path', '')
        file_name = getattr(element, 'name', 'inconnu')
        mime_type = getattr(element, 'mime', '')
        file_ext = os.path.splitext(file_path)[1].lower() if file_path else ''

        try:
            if "image" in mime_type or file_ext in ['.jpg', '.jpeg', '.png', '.svg', '.gif']:
                await process_image_file(element)
                continue

            if file_ext not in SUPPORTED_FILE_TYPES:
                await cl.Message(content=f"❌ Format de fichier non supporté: **{file_ext}** pour '{file_name}'. Veuillez uploader un fichier pris en charge.").send()
                logger.warning(f"Unsupported file format: {file_ext} for {file_name}.")
                continue

            await process_standard_file(element, file_ext, file_name)

        except Exception as e:
            await cl.Message(content=f"❌ Erreur lors du traitement du fichier '{file_name}': {str(e)}. Veuillez réessayer ou vérifier le fichier.").send()
            logger.exception(f"Error in handle_attachments for file {file_name}.")

async def process_image_file(element: cl.File):
    await cl.Message(content=f"🖼️ Analyse de l'image '{element.name}' en cours...").send()
    logger.info(f"Starting image processing for: {element.name}")
    try:
        if not _image_analyzer:
            raise ValueError("ImageAnalyzer not initialized in file_handlers.")
        if not _rag_system:
            raise ValueError("RAGSystem not initialized in file_handlers.")

        processed_data = _image_analyzer.process_image_complete(element.path)

        if processed_data['status'] == 'error':
            await cl.Message(content=f"❌ Échec de l'analyse de l'image '{element.name}': {processed_data['error']}").send()
            logger.warning(f"Image analysis failed for {element.name}: {processed_data['error']}.")
            return

        summary = processed_data['summary']
        full_analysis = processed_data['analysis']

        identified_country = "Non identifié"
        is_flag_classification = any("flag" in c['label'].lower() for c in full_analysis.get('classifications', [])[:2])

        if is_flag_classification:
            await cl.Message(content=f"🤔 Tentative d'identification du pays à partir de l'image...").send()
            try:
                # MODIFICATION ICI : Utilisez "gemini-pro-vision" pour l'identification d'image
                llm_for_identification = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.1, google_api_key=Config.GOOGLE_API_KEY)

                identification_prompt = f"""Vous êtes un expert en drapeaux. En vous basant **uniquement** sur la description fournie de l'image, nommez le pays correspondant à ce drapeau.

                Description de l'image:
                {summary}

                Si la description correspond clairement à un drapeau, répondez uniquement avec le nom du pays.
                Si la description ne suffit pas pour identifier un pays avec certitude, répondez "Pays non identifiable avec les informations fournies."

                Exemples:
                - Description: "Description: a flag with three horizontal stripes of black, red, and yellow" -> Allemagne
                - Description: "Description: a vertical tricolor flag with green, white, and red stripes" -> Italie
                - Description: "Description: a red banner with a white circle and a red sun" -> Japon

                Quel est le pays correspondant à ce drapeau ?
                Réponse:"""

                llm_response = await llm_for_identification.ainvoke(identification_prompt)
                llm_identified_country = llm_response.content.strip()

                if "non identifiable" not in llm_identified_country.lower() and llm_identified_country:
                    identified_country = llm_identified_country
                    await cl.Message(content=f"✅ Pays identifié: **{identified_country}**.").send()
                else:
                    await cl.Message(content=f"ℹ️ Le pays n'a pas pu être identifié avec certitude à partir de la description du drapeau. ({llm_identified_country})").send()

            except Exception as llm_e:
                logger.error(f"Erreur lors de l'identification du pays par LLM: {llm_e}", exc_info=True)
                await cl.Message(content=f"❌ Erreur lors de l'identification du pays par l'IA: {str(llm_e)}").send()

        doc_content = (f"Résumé de l'image : {summary}\n"
                       f"Description détaillée: {full_analysis.get('description', 'N/A')}\n"
                       f"Classifications: {', '.join([c['label'] for c in full_analysis.get('classifications', [])])}\n"
                       f"Couleurs: {full_analysis.get('color_analysis', {}).get('dominant_colors', 'N/A')}\n"
                       f"Pays identifié (si applicable): {identified_country}")

        doc_ids = _rag_system.add_document(
            content=doc_content,
            metadata={
                "source": element.name,
                "type": "image_analysis",
                "original_summary": summary,
                "image_classifications": [c['label'] for c in full_analysis.get('classifications', [])],
                "image_description": full_analysis.get('description', ''),
                "image_dominant_colors": full_analysis.get('color_analysis', {}).get('dominant_colors', ''),
                "identified_country": identified_country,
                "uuid": str(uuid.uuid4())
            }
        )
        if not doc_ids:
            await cl.Message(content=f"⚠️ Le fichier '{element.name}' n'a pas été indexé car aucun contenu utile n'a été généré ou un problème est survenu lors de l'ajout à l'index.").send()
            logger.warning(f"No document IDs returned for {element.name}. Content might have been empty or indexing failed.")
            return

        await cl.Message(content=f"✅ Image '{element.name}' analysée et indexée (ID: `{doc_ids[0] if doc_ids else 'N/A'}`). Résumé: {summary[:100]}... Pays: {identified_country}").send()
        logger.info(f"Image '{element.name}' analyzed and indexed successfully. IDs: {doc_ids}")
    except Exception as e:
        await cl.Message(content=f"❌ Erreur traitement image '{element.name}': {str(e)}. Vérifiez votre `image_analyzer.py` et les dépendances.").send()
        logger.exception(f"Error processing image {element.name}.")
        return