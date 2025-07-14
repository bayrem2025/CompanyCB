from universal_processor import UniversalExtractor
from vectordb import create_vectorstore, load_vectorstore, vectorstore_exists
from rag import setup_qa_chain
import chainlit as cl
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@cl.on_chat_start
async def initialize():
    """
    Initializes the chat session by loading or creating a vectorstore
    and setting up the QA chain.
    """
    try:
        if vectorstore_exists():
            vectorstore = load_vectorstore()
            msg = "Existing knowledge base loaded."
        else:
            logger.info("Creating a new knowledge base...")
            documents = UniversalExtractor.process_directory()
            
            if not documents:
                raise ValueError("No valid files found in the data/ folder.")
                
            logger.info(f"Number of documents to index: {len(documents)}")
            vectorstore = create_vectorstore(documents)
            msg = f"Knowledge base created with {len(documents)} chunks."
            
            # Log processed file types
            file_types = set(doc.metadata.get('file_type', 'unknown') for doc in documents)
            logger.info(f"Processed file types: {', '.join(file_types)}")

        qa_chain = setup_qa_chain(vectorstore)
        cl.user_session.set("vectorstore", vectorstore)
        cl.user_session.set("qa_chain", qa_chain)
        cl.user_session.set("doc_type", "pdf")  # Default type
        cl.user_session.set("language", "en") # Default language to English
        
        await cl.Message(content=msg).send()
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        await cl.Message(content=f"Initialization error: {str(e)}").send()
        raise

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handles incoming messages, processes commands, and generates responses.
    """
    # Command to change document type
    if message.content.startswith("/type "):
        new_type = message.content.split()[1].lower()
        valid_types = ["pdf", "excel", "docx", "ppt", "txt", "csv"]
        
        if new_type in valid_types:
            cl.user_session.set("doc_type", new_type)
            await cl.Message(content=f"Document type changed to {new_type}.").send()
        else:
            await cl.Message(content=f"Invalid type. Options: {', '.join(valid_types)}.").send()
        return

    # Command to change response language
    if message.content.startswith("/lang "):
        new_lang = message.content.split()[1].lower()
        valid_langs = ["en", "fr"]
        if new_lang in valid_langs:
            cl.user_session.set("language", new_lang)
            await cl.Message(content=f"Response language set to {new_lang}.").send()
        else:
            await cl.Message(content=f"Invalid language. Options: {', '.join(valid_langs)}.").send()
        return

    vectorstore = cl.user_session.get("vectorstore")
    qa_chain = cl.user_session.get("qa_chain")
    doc_type = cl.user_session.get("doc_type", "pdf")
    response_language = cl.user_session.get("language", "en") # Get current language setting
    
    # Recreate the chain with the correct document type and language
    qa_chain = setup_qa_chain(vectorstore, doc_type, response_language)
    
    # Call with the new structure
    response = await qa_chain.ainvoke(
        {"input": message.content},
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    
    # Access the correct keys
    answer = response["answer"]
    source_docs = response["context"]
    
    sources = {}
    for doc in source_docs:
        src = doc.metadata["source"]
        file_type = doc.metadata.get("file_type", "unknown")
        ref = f"chunk {doc.metadata.get('chunk', '?')}"
        
        if file_type == "pptx":
            ref = f"slide {doc.metadata.get('slide', '?')}"
        elif file_type in ["xlsx", "xls", "csv"]:
            ref = f"row {doc.metadata.get('row', '?')}"
            
        sources.setdefault(src, set()).add(ref)
    
    if sources:
        answer += "\n\nSources:\n" + "\n".join(
            f"- {os.path.basename(src)} ({', '.join(sorted(refs))})"
            for src, refs in sources.items()
        )
    
    await cl.Message(content=answer).send()
