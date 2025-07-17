
# clear_db.py
from vectordb import clear_vectorstore

if __name__ == "__main__":
    if clear_vectorstore():
        print("✅ Vectorstore cleared successfully!")
    else:
        print("❌ Failed to clear vectorstore. See error above.")