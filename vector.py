import os
import faiss
import numpy as np
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

class PDFVectorDatabase:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.texts = []

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return texts

    def add_pdf(self, pdf_path: str):
        texts = self.extract_text_from_pdf(pdf_path)
        embeddings = self.model.encode(texts)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)

    def search(self, query: str, top_k: int = 5):
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec, dtype=np.float32), top_k)
        results = []
        for idx in I[0]:
            results.append(self.texts[idx])
        return results

# Exemple d'utilisation :
# db = PDFVectorDatabase()
# db.add_pdf("mon_fichier.pdf")
# print(db.search("ma question"))