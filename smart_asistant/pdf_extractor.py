import os
import glob
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Chemin vers le dossier contenant les PDF
PDF_FOLDER = "pdfs/file.pdf"
VECTOR_DB_PATH = "vector_db.index"

# 1. Extraction du texte des fichiers PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# 2. Embedding des textes avec un modèle de phrase
def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True)

# 3. Création de la base de données vectorielle FAISS
def build_vector_db(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    # Charger le modèle d'embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extraire les textes de tous les PDF
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    documents = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        documents.append(text)

    # Embedding des documents
    embeddings = embed_texts(documents, model)
    embeddings = np.array(embeddings).astype("float32")

    # Création et sauvegarde de la base de données vectorielle
    index = build_vector_db(embeddings)
    faiss.write_index(index, VECTOR_DB_PATH)

    print(f"Base de données vectorielle créée avec {len(documents)} documents.")