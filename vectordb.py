from langchain_community.embeddings import OllamaEmbeddings  # Revert to OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import VECTORDB_DIR, EMBEDDING_MODEL
import os
import shutil
from langchain.schema import Document

def create_vectorstore(documents: list[Document]):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)  # Use OllamaEmbeddings
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORDB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vectorstore

def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )

def vectorstore_exists():
    db_file = os.path.join(VECTORDB_DIR, "chroma.sqlite3")
    return os.path.exists(db_file)

def get_vectorstore_stats():
    if not vectorstore_exists():
        return {"status": "Vectorstore not initialized"}
    vectorstore = load_vectorstore()
    collection = vectorstore._collection
    return {
        "document_count": collection.count(),
        "metadata_fields": list(collection.get(include=["metadatas"])["metadatas"][0].keys()),
        "embedding_model": EMBEDDING_MODEL
    }

def clear_vectorstore():
    """Completely clear the vector store"""
    if os.path.exists(VECTORDB_DIR):
        try:
            shutil.rmtree(VECTORDB_DIR)
            os.makedirs(VECTORDB_DIR, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error clearing vectorstore: {e}")
            return False
    return True