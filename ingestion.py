import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DIR, EMBED_MODEL

def ingest_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        return "PDF file not found."

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTOR_DIR)

    return f"Indexed {len(chunks)} chunks successfully."
