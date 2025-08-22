# here you can see my project demo video



https://github.com/user-attachments/assets/5768c9f0-7905-4a8f-b227-3487bac54008



# 📚 RAG-PDF-Gradio

A **local PDF chatbot** using **LangChain, FAISS, HuggingFace embeddings, Groq LLM**, and **Gradio UI**. Users can upload PDFs, ask questions, and get answers with references from the PDF.

---

## Features

- Upload PDFs and create embeddings using **FAISS**.
- Ask questions and get **context-aware answers** from the PDF.
- Answers include **references** (page numbers or section names).
- Maintains **conversation memory**.
- Clean **prompt engineering** for accurate responses.

---



---

## Requirements

- Python 3.10+
- pip

Dependencies (see `requirements.txt`):


---

## Setup & Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Jakaria6284/chat_with_pdf_langchain.git
cd chat_with_pdf_langchain


python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

pip install -r requirements.txt


GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" #you can make api key from this website https://groq.com/


python app.py  # paste in into your terminal that run the gradio interface after that you can test the system
