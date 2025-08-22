import os

# ---- Paths ----
DATA_DIR = "data"
VECTOR_DIR = "vectorstore/faiss"
UPLOADED_PDF_PATH = os.path.join(DATA_DIR, "uploaded.pdf")

# ---- Models ----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# ---- API Keys ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
