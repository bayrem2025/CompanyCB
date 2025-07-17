import os
import uuid
import logging
from typing import Optional, List

import chainlit as cl
import torch # Required for ImageAnalyzer
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from duckduckgo_search import DDGS

from config import Config
from utils.file_utils import (
    extract_text_from_txt,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_excel,
    extract_text_from_pptx
)

# NEW: Import your ImageAnalyzer
from image_analyzer import ImageAnalyzer

from rag_system import MultimodalRAGSystem

# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
rag_system: Optional[MultimodalRAGSystem] = None
image_analyzer: Optional[ImageAnalyzer] = None # NEW: Global instance for ImageAnalyzer

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

@cl.on_chat_start
async def init_rag_system():
    global rag_system, image_analyzer # NEW: Initialize image_analyzer
    try:
        Config.create_directories()
        await cl.Message(content="🚀 Initialisation du système RAG multimodal...").send()

        rag_system = MultimodalRAGSystem()

        # NEW: Initialize ImageAnalyzer
        await cl.Message(content="🧠 Chargement des modèles d'analyse d'image (BLIP, ViT, OpenCV)...").send()
        image_analyzer = ImageAnalyzer()
        await cl.Message(content="✅ Modèles d'analyse d'image chargés.").send()

        await cl.Message(content="✅ Système prêt à traiter documents et images.").send()
    except Exception as e:
        await cl.Message(content=f"❌ Échec de l'initialisation du système: {str(e)}. Veuillez vérifier les configurations et assurez-vous que les modèles d'AI sont accessibles.").send()
        logger.exception("Failed to initialize RAG system.")
        raise

# -------------------------------------------------------------
# --- START OF FILE/IMAGE PROCESSING HELPER FUNCTIONS ---
# These must be defined before they are called by handle_attachments and process_message
# -------------------------------------------------------------

async def process_image_file(element: cl.File):
    await cl.Message(content=f"🖼️ Analyse de l'image '{element.name}' en cours...").send()
    logger.info(f"Starting image processing for: {element.name}")
    try:
        # Use ImageAnalyzer's process_image_complete
        processed_data = image_analyzer.process_image_complete(element.path) # Pass the file path

        if processed_data['status'] == 'error':
            await cl.Message(content=f"❌ Échec de l'analyse de l'image '{element.name}': {processed_data['error']}").send()
            logger.warning(f"Image analysis failed for {element.name}: {processed_data['error']}.")
            return

        summary = processed_data['summary'] # e.g., "Description: the flag of italy | Catégorie principale: flagpole, flagstaff (63.31%) | Dimensions: 1200x800 px | Couleurs dominantes: 5 principales..."
        full_analysis = processed_data['analysis']

        # NEW: Try to identify the country if it's a flag
        identified_country = "Non identifié"
        # Check if the primary classification suggests it's a flag or similar
        # Consider top 2 classifications for keyword "flag"
        is_flag_classification = any("flag" in c['label'].lower() for c in full_analysis.get('classifications', [])[:2])

        if is_flag_classification:
            await cl.Message(content=f"🤔 Tentative d'identification du pays à partir de l'image...").send()
            try:
                # Use a specific prompt for the LLM to identify the country from the description
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


        # Combine analysis and identified country into the content for RAG
        doc_content = f"Résumé de l'image : {summary}\n" \
                      f"Description détaillée: {full_analysis.get('description', 'N/A')}\n" \
                      f"Classifications: {', '.join([c['label'] for c in full_analysis.get('classifications', [])])}\n" \
                      f"Couleurs: {full_analysis.get('color_analysis', {}).get('dominant_colors', 'N/A')}\n" \
                      f"Pays identifié (si applicable): {identified_country}" # Add identified country to content

        doc_ids = rag_system.add_document( # add_document returns a list of IDs
            content=doc_content, # Store the detailed description for better RAG
            metadata={
                "source": element.name,
                "type": "image_analysis", # Changed type to be more specific
                "original_summary": summary, # Keep the generated summary for easy access
                "image_classifications": [c['label'] for c in full_analysis.get('classifications', [])],
                "image_description": full_analysis.get('description', ''),
                "image_dominant_colors": full_analysis.get('color_analysis', {}).get('dominant_colors', ''),
                "identified_country": identified_country, # Also store in metadata
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

        doc_ids = rag_system.add_document( # add_document returns a list of IDs
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

# -------------------------------------------------------------
# --- END OF FILE/IMAGE PROCESSING HELPER FUNCTIONS ---
# -------------------------------------------------------------

@cl.on_message
async def process_message(message: cl.Message):
    global rag_system

    if not message.content and not getattr(message, 'elements', []):
        await cl.Message(content="❌ Veuillez poser une question ou envoyer un fichier à analyser.").send()
        return

    # Call handle_attachments FIRST if there are elements (this was the NameError fix)
    if hasattr(message, 'elements') and message.elements:
        await handle_attachments(message.elements)

    # Then handle the text query if present
    if message.content and message.content.strip():
        await handle_text_query(message.content)

async def perform_web_search(query: str, llm):
    if cl.user_session.get("skip_web_search", False):
        return

    await cl.Message(content="🌐 Exécution de la recherche web via DuckDuckGo...").send()
    logger.info(f"Performing web search for query: '{query}'")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(keywords=query, max_results=3)

        search_results_text = []
        if results:
            for i, item in enumerate(results, 1):
                title = item.get('title', 'Titre inconnu')
                href = item.get('href', 'Lien inconnu')
                body = item.get('body', 'Extrait non disponible')
                search_results_text.append(f"**Résultat {i}:**\n**Titre:** {title}\n**Lien:** {href}\n**Extrait:** {body}\n---")

        if search_results_text:
            web_context = "\n\n".join(search_results_text)
            web_prompt = f"""Vous êtes un assistant qui utilise des informations externes pour répondre.
Question de l'utilisateur: {query}

Informations trouvées via la recherche web:
{web_context}

Utilisez ces informations pour répondre à la question de manière concise et utile. Si les informations ne sont pas suffisantes, indiquez-le clairement."""

            web_llm_response = await llm.ainvoke(web_prompt)
            await cl.Message(content=f"**Réponse (via Recherche Web) :**\n{web_llm_response.content.strip()}").send()
            logger.info("Web search completed and response sent.")
        else:
            await cl.Message(content="🤷‍♂️ Désolé, je n'ai pas trouvé d'informations pertinentes ni dans mes documents, ni via la recherche web pour cette question.").send()
            logger.info("No web search results found.")

    except Exception as web_e:
        await cl.Message(content=f"❌ Erreur lors de la recherche web : {str(web_e)}. Le service de recherche est peut-être temporairement indisponible ou votre requête est mal formée.").send()
        logger.exception(f"Error during web search for query: '{query}'.")

async def handle_text_query(query: str):
    if not query.strip():
        return

    await cl.Message(content=f"🔍 Recherche de réponse pour: '{query}' en cours...").send()
    logger.info(f"Handling text query: '{query}'")

    try:
        llm = None
        if Config.GROQ_API_KEY:
            llm = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                temperature=Config.TEMPERATURE
            )
            logger.info(f"Using Groq LLM: {Config.GROQ_MODEL}")
        else:
            llm = ChatGoogleGenerativeAI(model=Config.GOOGLE_GEMINI_LLM_MODEL, temperature=Config.TEMPERATURE, google_api_key=Config.GOOGLE_API_KEY)
            logger.info(f"Using Google Gemini LLM: {Config.GOOGLE_GEMINI_LLM_MODEL}")


        flag_keywords = ["pays", "drapeau", "nationalité", "de qui", "quel pays", "quel est le pays"] # Added "quel pays", "quel est le pays"
        uml_keywords = ["méthodes", "operations", "classe", "uml", "diagramme", "attributs"]

        is_flag_query = any(keyword in query.lower() for keyword in flag_keywords)
        is_uml_query = any(keyword in query.lower() for keyword in uml_keywords)

        n_results_for_search = 5 # Retrieve more, then filter

        results = rag_system.search(query, n_results=n_results_for_search)
        logger.info(f"Initial RAG search returned {len(results)} results.")

        filtered_results = []

        # NOTE: The type for image summaries is now 'image_analysis' from image_analyzer.py
        if is_flag_query:
            image_summaries = [r for r in results if r['metadata'].get('type') == 'image_analysis']
            if image_summaries:
                image_summaries.sort(key=lambda x: x['score'], reverse=True)
                # For flag queries, if there's an image, assume it's the primary context
                filtered_results = [image_summaries[0]]
                logger.info(f"Flag query: Filtered to top image analysis: {filtered_results[0]['metadata'].get('source')}")
            else:
                logger.warning(f"Flag query but no 'image_analysis' found in top {n_results_for_search} results. Query: '{query}'")
                # Fallback: if no image analysis, allow general RAG results or web search
                filtered_results = results # Or, you could explicitly only allow text docs here
        elif is_uml_query:
            uml_summaries = [r for r in results if r['metadata'].get('type') == 'image_analysis' and ("diagramme uml" in r['content'].lower() or "classe" in r['content'].lower() or "méthodes" in r['content'].lower() or "operations" in r['content'].lower())]
            if uml_summaries:
                uml_summaries.sort(key=lambda x: x['score'], reverse=True)
                filtered_results = [uml_summaries[0]]
                logger.info(f"UML query: Filtered to top UML image analysis: {filtered_results[0]['metadata'].get('source')}")
            else:
                logger.warning(f"UML query but no relevant 'image_analysis' found. Query: '{query}'")
                filtered_results = [r for r in results if r['metadata'].get('type') in ['txt', 'pdf', 'docx', 'pptx', 'xlsx', 'csv']]
        else:
            filtered_results = results
            logger.info(f"General query: Using all {len(filtered_results)} initial RAG results.")


        rag_response_content = None
        if filtered_results:
            context = "\n".join(
                f"Source: {res['metadata'].get('source', 'Inconnu')} (Type: {res['metadata'].get('type', 'document')})\nContenu: {res['content']}"
                for res in filtered_results
            )
            logger.debug(f"Context passed to LLM:\n{context}")

            if is_flag_query:
                # The LLM's role is to confirm the identified country or interpret the full description
                rag_prompt = f"""Vous êtes un assistant expert en drapeaux. Votre tâche est de répondre à la question de l'utilisateur en utilisant **exclusivement** les "Documents pertinents" ci-dessous.

                **Instructions essentielles:**
                - La question de l'utilisateur est : "{query}".
                - Les "Documents pertinents" peuvent contenir une description détaillée d'un drapeau, des classifications d'images, et potentiellement un pays déjà identifié par une analyse préliminaire.
                - Si un "Pays identifié" est mentionné dans le document et semble pertinent, utilisez-le comme réponse principale.
                - Si aucun pays n'est explicitement identifié mais la description du drapeau est claire (couleurs, motif, orientation), utilisez cette description pour déduire le pays.
                - Si les informations sont insuffisantes pour identifier un pays, indiquez-le clairement.

                Documents pertinents trouvés dans la base de connaissances:
                {context}

                Quel est le pays du drapeau ? Votre réponse doit être concise et le nom du pays ou une déclaration d'information insuffisante :"""
            elif is_uml_query:
                rag_prompt = f"""Vous êtes un assistant expert en diagrammes UML. Votre tâche est d'extraire les informations pertinentes sur les **méthodes (opérations)** des classes spécifiées dans les "Documents pertinents" ci-dessous.

                **Instructions:**
                - La question de l'utilisateur est : "{query}".
                - Les "Documents pertinents" contiennent une description textuelle d'un diagramme de classes UML, incluant les classes, leurs attributs et leurs méthodes/opérations, générée par une analyse d'image.
                - Votre réponse doit lister les méthodes/opérations pour chaque classe pertinente mentionnée dans la question ou pour toutes les classes si la question est générale.
                - Pour chaque méthode, incluez son nom et sa signature (paramètres, type de retour) si disponible.
                - Si la question porte sur une classe spécifique (ex: "méthodes de la classe Employee"), ne donnez que les méthodes de cette classe.
                - Si une méthode n'a pas de paramètres ou de type de retour, indiquez-le clairement (ex: "méthode(): aucun paramètre, aucun retour").
                - Si aucune méthode n'est trouvée pour la classe demandée ou dans le diagramme général, indiquez-le.

                Documents pertinents trouvés dans la base de connaissances:
                {context}

                Veuillez répondre de manière structurée en listant les méthodes, ou indiquez si l'information est absente :"""
            else: # General query prompt
                rag_prompt = f"""Vous êtes un assistant qui répond aux questions en se basant strictement sur les documents fournis.

                Question de l'utilisateur: {query}

                Documents pertinents trouvés dans la base de connaissances:
                {context}

                Veuillez répondre à la question de manière concise et précise **en utilisant uniquement les informations contenues dans les documents ci-dessus**.
                Si les informations des documents ne sont pas suffisantes pour répondre complètement à la question, veuillez indiquer clairement "Je n'ai pas trouvé d'informations suffisantes dans les documents fournis pour répondre à cette question.":"""

            rag_llm_response = await llm.ainvoke(rag_prompt)
            rag_response_content = rag_llm_response.content.strip()
            logger.info(f"LLM generated RAG response: {rag_response_content[:200]}...")

            insufficient_keywords = [
                "je n'ai pas trouvé d'informations suffisantes",
                "les documents fournis ne contiennent pas",
                "pas d'informations pertinentes",
                "ne contient pas d'informations",
                "ne mentionne pas",
                "informations non disponibles",
                "non identifiable" # Added for flag context
            ]

            rag_found_sufficient_info = True
            for keyword in insufficient_keywords:
                if keyword in rag_response_content.lower():
                    rag_found_sufficient_info = False
                    break

            if rag_found_sufficient_info:
                await cl.Message(content=f"**Réponse des documents:**\n{rag_response_content}").send()
            else:
                cl.user_session.set("skip_web_search", False)
                no_web_search_queries = ["combien de produit", "nombre de produits", "quantité de produits"]

                if query.strip().lower() in no_web_search_queries:
                    cl.user_session.set("skip_web_search", True)
                    await cl.Message(content=f"**Réponse des documents (informations limitées):**\n{rag_response_content}\n\n🤷‍♂️ Je n'ai pas trouvé de données spécifiques sur le nombre de produits dans mes documents pour répondre à cette question.").send()
                else:
                    await cl.Message(content=f"**Réponse des documents (informations limitées):**\n{rag_response_content}\n\n🌐 Les documents internes n'ont pas fourni de réponse complète. Tentative de recherche web...").send()
                    await perform_web_search(query, llm)
        else:
            logger.info(f"No filtered RAG results for query: '{query}'. Proceeding to web search if applicable.")
            cl.user_session.set("skip_web_search", False)
            no_web_search_queries = ["combien de produit", "nombre de produits", "quantité de produits"]

            if query.strip().lower() in no_web_search_queries:
                cl.user_session.set("skip_web_search", True)
                await cl.Message(content="🔍 Aucun résultat pertinent trouvé dans vos documents indexés.").send()
                await cl.Message(content="🤷‍♂️ Je n'ai pas trouvé de données spécifiques sur le nombre de produits dans mes documents pour répondre à cette question.").send()
            else:
                await cl.Message(content="🔍 Aucun résultat pertinent trouvé dans vos documents indexés.").send()
                await cl.Message(content="🌐 Les documents internes n'ont pas suffi. Tentative de recherche web...").send()
                await perform_web_search(query, llm)

    except Exception as e:
        await cl.Message(content=f"❌ Erreur lors du traitement de la requête: {str(e)}. Veuillez vérifier les configurations de votre LLM (Groq/Google Gemini) et votre GOOGLE_API_KEY.").send()
        logger.exception(f"Error handling text query: '{query}'.")

if __name__ == "__main__":
    cl.run()