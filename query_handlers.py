import logging
from typing import Optional, List, Dict, Any

import chainlit as cl
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS

from config import Config
from rag_system import MultimodalRAGSystem # Importation de MultimodalRAGSystem

logger = logging.getLogger(__name__)

# Global instance for RAG system, will be set from app.py
_rag_system: Optional[MultimodalRAGSystem] = None

def set_rag_system_instance(rag_sys: MultimodalRAGSystem):
    global _rag_system
    _rag_system = rag_sys
    logger.info("RAG system instance set in query_handlers.")

async def perform_web_search(query: str, llm):
    """
    Performs a web search using DuckDuckGo and sends the results to the user.
    """
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
    """
    Handles text queries, performing RAG search and potentially web search.
    """
    if not query.strip():
        return

    await cl.Message(content=f"🔍 Recherche de réponse pour: '{query}' en cours...").send()
    logger.info(f"Handling text query: '{query}'")

    try:
        if not _rag_system:
            raise ValueError("RAG system not initialized in query_handlers.")

        llm = None
        if Config.GROQ_API_KEY:
            llm = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                temperature=Config.TEMPERATURE
            )
            logger.info(f"Using Groq LLM: {Config.GROQ_MODEL}")
        elif Config.GOOGLE_API_KEY: # Assurez-vous d'avoir une clé Google pour Gemini
            # Utilisation du modèle défini dans config.py pour les requêtes textuelles
            llm = ChatGoogleGenerativeAI(
                model=Config.GOOGLE_GEMINI_LLM_MODEL,
                temperature=Config.TEMPERATURE,
                google_api_key=Config.GOOGLE_API_KEY
            )
            logger.info(f"Using Google Gemini LLM: {Config.GOOGLE_GEMINI_LLM_MODEL}")
        else:
            await cl.Message(content="❌ Aucune clé API LLM (Groq ou Google Gemini) n'est configurée. Impossible de traiter la requête.").send()
            logger.error("Neither GROQ_API_KEY nor GOOGLE_API_KEY is set. Cannot process query.")
            return

        flag_keywords = ["pays", "drapeau", "nationalité", "de qui", "quel pays", "quel est le pays"]
        uml_keywords = ["méthodes", "operations", "classe", "uml", "diagramme", "attributs"]

        is_flag_query = any(keyword in query.lower() for keyword in flag_keywords)
        is_uml_query = any(keyword in query.lower() for keyword in uml_keywords)

        n_results_for_search = 5

        results = _rag_system.search(query, n_results=n_results_for_search)
        logger.info(f"Initial RAG search returned {len(results)} results.")

        filtered_results = []

        if is_flag_query:
            image_summaries = [r for r in results if r['metadata'].get('type') == 'image_analysis']
            if image_summaries:
                image_summaries.sort(key=lambda x: x['score'], reverse=True)
                filtered_results = [image_summaries[0]]
                logger.info(f"Flag query: Filtered to top image analysis: {filtered_results[0]['metadata'].get('source')}")
            else:
                logger.warning(f"Flag query but no 'image_analysis' found in top {n_results_for_search} results. Query: '{query}'")
                filtered_results = results # Fallback to all results if no image analysis
        elif is_uml_query:
            uml_summaries = [r for r in results if r['metadata'].get('type') == 'image_analysis' and ("diagramme uml" in r['content'].lower() or "classe" in r['content'].lower() or "méthodes" in r['content'].lower() or "operations" in r['content'].lower())]
            if uml_summaries:
                uml_summaries.sort(key=lambda x: x['score'], reverse=True)
                filtered_results = [uml_summaries[0]]
                logger.info(f"UML query: Filtered to top UML image analysis: {filtered_results[0]['metadata'].get('source')}")
            else:
                logger.warning(f"UML query but no relevant 'image_analysis' found. Query: '{query}'")
                # Fallback to general documents if no specific UML image analysis
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
                "non identifiable"
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