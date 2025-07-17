import os
import logging
import chainlit as cl
import langdetect
from universal_processor import UniversalExtractor
from vectordb import create_vectorstore, load_vectorstore, vectorstore_exists
from rag import setup_qa_chain

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def initialize_knowledge_base():
    try:
        # UI settings
        css = """
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
        }
        .message {
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
        }
        .user-message {
            background-color: #0000FF;
            align-self: flex-end;
        }
        .bot-message {
            background-color: #f8f9fa;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .sources {
            font-size: 0.85em;
            color: #666;
            margin-top: 10px;
            border-top: 1px solid #eee;
            padding-top: 8px;
        }
        """
        await cl.ChatSettings([
            cl.input_widget.Select(
                id="DocType", label="Document Type",
                values=["pdf", "excel", "docx", "ppt", "txt", "csv"], initial_value="pdf"
            ),
            cl.input_widget.Switch(id="Detailed", label="Detailed Answers", initial_value=False)
        ]).send()

        # Load or create vectorstore
        if vectorstore_exists():
            vectorstore = load_vectorstore()
            msg = "🔍 Existing knowledge base loaded successfully."
        else:
            documents = UniversalExtractor.process_directory()
            if not documents:
                raise ValueError("No valid files found in the data/ folder.")
            vectorstore = create_vectorstore(documents)
            msg = f"📚 Knowledge base created with {len(documents)} document chunks."

        # Store in session
        qa_chain = setup_qa_chain(vectorstore)
        cl.user_session.set("vectorstore", vectorstore)
        cl.user_session.set("qa_chain", qa_chain)
        cl.user_session.set("doc_type", "pdf")
        cl.user_session.set("language", "en")

        await cl.Message(content=f"## 🤖 Document AI Assistant\n\n{msg}").send()

    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        await cl.Message(content=f"⚠️ Initialization error: {e}").send()
        raise

async def process_user_message(message: cl.Message):
    if message.content.startswith("/type "):
        new_type = message.content.split()[1].lower()
        valid_types = ["pdf", "excel", "docx", "ppt", "txt", "csv"]
        if new_type in valid_types:
            cl.user_session.set("doc_type", new_type)
            await cl.Message(content=f"✅ Document type changed to {new_type}.").send()
        else:
            await cl.Message(content=f"❌ Invalid type. Options: {', '.join(valid_types)}.").send()
        return

    await cl.Message(content="💭 Chat is thinking...").send()

    try:
        lang = langdetect.detect(message.content)
        lang = lang if lang in ["en", "fr"] else "en"
    except:
        lang = "en"

    vectorstore = cl.user_session.get("vectorstore")
    doc_type = cl.user_session.get("doc_type", "pdf")
    qa_chain = setup_qa_chain(vectorstore, doc_type, lang)

    response = await qa_chain.ainvoke(
        {"input": message.content},
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    answer = response["answer"]
    context = response.get("context", [])
    sources_text = build_sources_html(context)
    await cl.Message(content=answer + sources_text).send()

def build_sources_html(docs):
    sources = {}
    for doc in docs:
        src = doc.metadata["source"]
        refs = sources.setdefault(src, set())
        file_type = doc.metadata.get("file_type", "unknown")
        if file_type == "pptx":
            refs.add(f"slide {doc.metadata.get('slide', '?')}")
        elif file_type in ["xlsx", "xls", "csv"]:
            refs.add(f"row {doc.metadata.get('row', '?')}")
        else:
            refs.add(f"chunk {doc.metadata.get('chunk', '?')}")
    if not sources:
        return ""
    return "\n\n<div class='sources'>🔍 Sources:\n" + "\n".join(
        f"- 📄 {os.path.basename(src)} ({', '.join(sorted(refs))})"
        for src, refs in sources.items()
    ) + "</div>"
