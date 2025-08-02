# study_buddy_app.py

import streamlit as st
import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq LLM
os.environ["GROQ_API_KEY"] = "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"  # Replace securely
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# ----------------------------- Utility Functions ----------------------------- #

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file format."

def split_into_sections(text):
    import re
    sections = []
    current_section = []
    for line in text.split('\n'):
        if re.match(r'^([A-Z\s]{5,}|.*:)$', line.strip()):
            if current_section:
                sections.append('\n'.join(current_section).strip())
                current_section = []
        current_section.append(line)
    if current_section:
        sections.append('\n'.join(current_section).strip())
    return sections

def get_dual_chunks_for_sections(sections):
    all_small_chunks, all_big_chunks = [], []
    for section in sections:
        length = len(section)
        if length < 1000:
            s = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            b = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        elif length < 3000:
            s = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            b = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        else:
            s = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            b = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        all_small_chunks.extend(s.split_text(section))
        all_big_chunks.extend(b.split_text(section))
    return all_small_chunks, all_big_chunks

def build_index(chunks):
    vectors = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, chunks

def retrieve_dual_context(question, small_chunks, big_chunks, small_index, big_index, top_k=3):
    q_vec = embed_model.encode([question])
    _, small_ids = small_index.search(np.array(q_vec), top_k)
    _, big_ids = big_index.search(np.array(q_vec), top_k)
    small_ctx = "\n".join([small_chunks[i] for i in small_ids[0]])
    big_ctx = "\n".join([big_chunks[i] for i in big_ids[0]])
    return small_ctx + "\n\n" + big_ctx

def ask_question(question, small_chunks, big_chunks, small_index, big_index):
    context = retrieve_dual_context(question, small_chunks, big_chunks, small_index, big_index)
    prompt = f"""
You are a helpful study assistant. Use the following context only.
If the answer is not in the context, reply: "Not in PDF".

Context:
{context}

Question: {question}
"""
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_summary(text):
    prompt = f"Summarize the following clearly and concisely:\n\n{text}"
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_flashcards(text):
    prompt = f"Generate 5 Q&A flashcards from this content:\n\n{text}"
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_mcqs(text, difficulty="medium"):
    prompt = f"""
Generate 5 multiple choice questions of {difficulty} difficulty based on this content:
{text}

Format:
Q1. ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: ...
"""
    return llm.invoke([HumanMessage(content=prompt)]).content

# ----------------------------- Streamlit UI ----------------------------- #
st.set_page_config(page_title="📚 Study Buddy App", layout="wide")
st.title("🧠 Study Buddy App 999")

if "text" not in st.session_state:
    st.session_state.text = None
    st.session_state.small_chunks = []
    st.session_state.big_chunks = []
    st.session_state.small_index = None
    st.session_state.big_index = None

uploaded_file = st.sidebar.file_uploader("Upload your study material", type=["pdf", "txt", "docx"])
if uploaded_file:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    text = extract_text(uploaded_file)
    sections = split_into_sections(text)
    small_chunks, big_chunks = get_dual_chunks_for_sections(sections)
    small_index, _ = build_index(small_chunks)
    big_index, _ = build_index(big_chunks)

    st.session_state.text = text
    st.session_state.small_chunks = small_chunks
    st.session_state.big_chunks = big_chunks
    st.session_state.small_index = small_index
    st.session_state.big_index = big_index

menu = st.sidebar.radio("Choose Feature", ["❓ Ask Questions", "📚 Summary", "🧠 Flashcards", "📝 MCQ Generator"])

if st.session_state.text:
    if menu == "❓ Ask Questions":
        st.subheader("Ask a question strictly from the document")
        q = st.text_input("Your Question")
        if st.button("Ask") and q:
            response = ask_question(q,
                st.session_state.small_chunks,
                st.session_state.big_chunks,
                st.session_state.small_index,
                st.session_state.big_index)
            st.write(response)

    elif menu == "📚 Summary":
        st.subheader("Document Summary")
        if st.button("Generate Summary"):
            summary = generate_summary(st.session_state.text)
            st.text_area("Summary", summary, height=300)

    elif menu == "🧠 Flashcards":
        st.subheader("Flashcards (Q&A pairs)")
        if st.button("Generate Flashcards"):
            cards = generate_flashcards(st.session_state.text)
            st.text_area("Flashcards", cards, height=300)

    elif menu == "📝 MCQ Generator":
        st.subheader("Practice Test (MCQ)")
        level = st.selectbox("Difficulty", ["easy", "medium", "hard"])
        if st.button("Generate MCQs"):
            mcqs = generate_mcqs(st.session_state.text, difficulty=level)
            st.text_area("MCQs", mcqs, height=400)
else:
    st.info("👈 Please upload a PDF, DOCX or TXT file to begin.")
