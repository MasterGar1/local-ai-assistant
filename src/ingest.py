import os
import fitz  # PyMuPDF

def extract_text_from_file(file_path: str) -> str:
    """Extracts text content from various file formats (TXT, MD, PY, PDF, etc.)."""
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == ".pdf":
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            print(f"[Ingest System] Error reading PDF file '{file_path}': {e}")
        return text
    else:
        # Try standard text decodings
        for encoding in ["utf-8", "latin1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        print(f"[Ingest System] Warning: Could not decode text file '{file_path}'.")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Splits a document text into overlapping chunks for semantic retrieval."""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += (chunk_size - overlap)
        
    return [c for c in chunks if c]
