import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "./data"  # Dossier contenant tous les fichiers
VECTORDB_DIR = "./vectordb"

# Sous-dossiers spécifiques par type de document
CSV_DIR = os.path.join(DATA_DIR, "csv")
DOCX_DIR = os.path.join(DATA_DIR, "docx")
EXCEL_DIR = os.path.join(DATA_DIR, "excel")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
PPT_DIR = os.path.join(DATA_DIR, "ppt")
TXT_DIR = os.path.join(DATA_DIR, "txt")

# Configuration du modèle
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"  # Changed to nomic-embed-text
TEMPERATURE = 0.3  # Lower value = more focused responses (0.1-0.5 recommended)

# Traitement de texte
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Paramètre spécifique pour Excel
MAX_ROWS_PER_CHUNK = 50
