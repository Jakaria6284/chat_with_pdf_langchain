from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import VECTOR_DIR, EMBED_MODEL, GROQ_API_KEY, GROQ_MODEL

# ---- Memory ----
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def reset_memory():
    """Clear previous conversation memory (use after ingesting a new PDF)"""
    memory.clear()

def normalize_question(question: str) -> str:
    """
    Ensure the question starts with 'What' if not already starting
    with Who/When/Where/Why/How/What.
    """
    question = question.strip()
    if question == "":
        return "What?"
    
    first_word = question.split()[0].lower()
    wh_words = ["what", "who", "when", "where", "why", "how","tell me about"]
    if first_word not in wh_words:
        question = "What " + question[0].lower() + question[1:] if len(question) > 0 else "What?"
    return question

def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

    # ---- System + Human Prompt with diverse few-shot examples ----
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

    chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a highly knowledgeable financial expert assistant. "
        "Use the context provided to answer the user question accurately. "
        "Always provide the answer in clear, concise sentences. "
        "Cite the reference (page number or section title) from the PDF where the information was found. "
        "If the answer is not present in the context, say 'The information is not available in the provided document.'"
    ),
    HumanMessagePromptTemplate.from_template(
        """Context:
{context}

User Question:
{question}

Instructions:
- Answer the question clearly and concisely.
- Always start the answer naturally (What, Who, When, Where, Why, How) depending on the question.
- Include the reference where the information is found in the format: Reference: page:X or section name.
- If the information is not in the context, state clearly that it is not available.

Answer:
"""
    )
])


    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )
    return chain

def answer(user_question: str) -> str:
    """
    Normalize the user's question and ask the RAG chain.
    """
    question = normalize_question(user_question)
    chain = build_chain()
    result = chain.invoke({"question": question})
    return result.get("answer") or result.get("result") or "No answer."
