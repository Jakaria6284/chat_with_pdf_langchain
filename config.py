import os

# ---- Paths that actually create the directory where actually store fiass vectore db and data(means pdf) in runtime when we upload pdf in our system ----
DATA_DIR = "data"
VECTOR_DIR = "vectorstore/faiss"
UPLOADED_PDF_PATH = os.path.join(DATA_DIR, "uploaded.pdf")

# ---- Models we are going to use in our system one for embedding and another for generation----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# ---- API Keys of our LLM model ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#ok our config is ready now go for ingestion part for text extreact split and convert it into vector
