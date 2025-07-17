import chromadb
from config import Config
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorDB:
    def __init__(self):
        Config.create_directories()
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=Config.GOOGLE_EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
        )
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_documents(self, documents):
        for doc in documents:
            doc_id = doc.metadata.get("source", str(id(doc)))
            embedding = self.embedding_model.embed_query(doc.page_content)
            self.collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[doc_id],
                embeddings=[embedding],
            )

    def similarity_search(self, query, top_k=5):
        embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return results["documents"][0] if results and results["documents"] else []
