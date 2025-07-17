import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # LLM Configuration (Prefer Groq, otherwise Google Gemini)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192") # Default Groq model

    # Google API Configuration (for Gemini Embeddings, Vision, and LLM if Groq not used)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_GEMINI_EMBEDDING_MODEL = os.getenv("GOOGLE_GEMINI_EMBEDDING_MODEL", "models/embedding-001") # New: Explicit Gemini embedding model
    GOOGLE_GEMINI_LLM_MODEL = os.getenv("GOOGLE_GEMINI_LLM_MODEL", "gemini-pro") # New: Explicit Gemini LLM model

    # ChromaDB Configuration
    CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

    # Document Processing Configuration
    DATA_PATH = os.getenv("DATA_PATH", "data")
    UPLOAD_PATH = os.getenv("UPLOAD_PATH", "uploads")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # General AI Settings
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

    @staticmethod
    def create_directories():
        """Creates necessary directories if they don't exist."""
        os.makedirs(Config.CHROMA_PATH, exist_ok=True)
        os.makedirs(Config.DATA_PATH, exist_ok=True)
        os.makedirs(Config.UPLOAD_PATH, exist_ok=True)
        logger.info(f"Vérification/création des répertoires: {Config.CHROMA_PATH}, {Config.DATA_PATH}, {Config.UPLOAD_PATH}")

    @staticmethod
    def validate_keys():
        """Validates that essential API keys are set."""
        if not Config.GROQ_API_KEY and not Config.GOOGLE_API_KEY:
            logger.warning("Neither GROQ_API_KEY nor GOOGLE_API_KEY is set. LLM functionality may be limited or unavailable.")
        
        if not Config.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY non trouvée. L'analyse d'images et les embeddings Gemini ne fonctionneront pas.")
        else:
            logger.info("GOOGLE_API_KEY trouvée.")
        
        logger.info("Configuration chargée.")

Config.validate_keys()