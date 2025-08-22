import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DIR, EMBED_MODEL 


#this method first load the pdf by using langchain document loader then
# recursively split the text and make document list 
# after that we convert row text into vector by our embedding model
#finally store the vector in fiass for symentic vector search
#fianlly save the fiass vectore db in local data folder that i mention in config folder


def ingest_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        return "PDF file not found."

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTOR_DIR)

    return f"Indexed {len(chunks)} chunks successfully."
