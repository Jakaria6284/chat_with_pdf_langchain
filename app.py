import os
import gradio as gr
from config import DATA_DIR, VECTOR_DIR, GROQ_API_KEY
from ingestion import ingest_pdf
from rag import answer, reset_memory

# ---- Runtime folders ----
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

def do_ingest(pdf_file):
    if GROQ_API_KEY.strip() == "":
        return "⚠️ GROQ_API_KEY is missing. Add it in your environment / HF Space secrets."
    if pdf_file is None:
        return "Please upload a PDF."

    # pdf_file is a path string from Gradio
    msg = ingest_pdf(pdf_file)
    reset_memory()
    return f"✅ {msg}"

def chat_fn(user_message, history):
    if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        return "Please upload a PDF and click 'Ingest PDF' first.", history

    ans = answer(user_message)
    history.append((user_message, ans))
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Chat with Your PDF — FAISS + Groq + Memory (LangChain)")

    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        ingest_btn = gr.Button("Ingest PDF")

    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat")
    msg = gr.Textbox(label="Ask a question about the PDF and press Enter")
    clear = gr.Button("Clear Chat")

    ingest_btn.click(do_ingest, inputs=pdf, outputs=status)
    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
