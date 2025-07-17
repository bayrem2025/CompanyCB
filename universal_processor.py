from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    TextLoader, ImageCaptionLoader
)

def load_and_process(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, autodetect_encoding=True)
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        loader = ImageCaptionLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return loader.load()
