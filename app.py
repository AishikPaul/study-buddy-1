# study_buddy_app.py

import streamlit as st
import os
import faiss
import numpy as np
import time
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

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
    return st.session_state.llm.invoke([HumanMessage(content=prompt)]).content

def generate_summary(text):
    prompt = f"Summarize the following clearly and concisely:\n\n{text}"
    return st.session_state.llm.invoke([HumanMessage(content=prompt)]).content

def generate_flashcards(text):
    prompt = f"Generate 5 Q&A flashcards from this content:\n\n{text}"
    return st.session_state.llm.invoke([HumanMessage(content=prompt)]).content

def generate_mcqs(text, difficulty="medium"):
    prompt = f"""
Generate exactly 5 multiple-choice questions of {difficulty} difficulty from the text below.
Each question should have four options (A, B, C, D) and the correct answer on a new line.

Use this format exactly:
Q1. Question text
A. Option A
B. Option B
C. Option C
D. Option D
Answer: B

Here is the content:
{text}
"""
    return st.session_state.llm.invoke([HumanMessage(content=prompt)]).content

def parse_mcqs(mcq_text):
    questions = []
    blocks = mcq_text.strip().split("Q")
    for block in blocks:
        if block.strip():
            lines = block.strip().split("\n")
            q = "Q" + lines[0]
            options = [line for line in lines[1:5] if line.strip().startswith(tuple("ABCD"))]
            answer_line = [line for line in lines if line.startswith("Answer")]
            ans = answer_line[0] if answer_line else "Answer:"
            if len(options) == 4:
                questions.append({"question": q, "options": options, "answer": ans})
    return questions

# ----------------------------- Streamlit UI ----------------------------- #
st.set_page_config(page_title="📚 Study Buddy App", layout="wide")
st.title("🧠 Study Buddy App")

api_key = st.sidebar.text_input("🔑 Enter your GROQ API Key", type="password")

if api_key:
    st.session_state.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=api_key)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if "text" not in st.session_state:
        st.session_state.text = None
        st.session_state.small_chunks = []
        st.session_state.big_chunks = []
        st.session_state.small_index = None
        st.session_state.big_index = None
        st.session_state.mcqs = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.user_answers = []
        st.session_state.time_limit = 30

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
            st.subheader("📝 Choose Mode for MCQs")
            mcq_mode = st.radio("Select Mode", ["📖 Q&A with Answers", "🧪 Practice Quiz"])
            level = st.selectbox("Difficulty", ["easy", "medium", "hard"])

            if mcq_mode == "🧪 Practice Quiz":
                num_q = st.slider("Number of Questions", 1, 10, 5)
                if st.button("Start Practice Quiz"):
                    mcq_text = generate_mcqs(st.session_state.text, difficulty=level)
                    st.session_state.mcqs = parse_mcqs(mcq_text)[:num_q]
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.user_answers = [None] * num_q

                if st.session_state.mcqs:
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        st.markdown("**Question Palette**")
                        for idx in range(len(st.session_state.mcqs)):
                            label = f"{idx + 1}"
                            color = "✅" if st.session_state.user_answers[idx] else "⬜"
                            if st.button(f"{color} {label}"):
                                st.session_state.quiz_index = idx

                    with col1:
                        i = st.session_state.quiz_index
                        q = st.session_state.mcqs[i]
                        st.write(f"**Q{i+1}. {q['question']}**")
                        choice = st.radio("Choose one:", q['options'], key=f"quiz_q_{i}", index=0)

                        cols = st.columns(3)
                        if cols[0].button("Save & Next"):
                            st.session_state.user_answers[i] = choice
                            correct = q['answer'].split(':')[-1].strip()
                            if choice.strip().startswith(correct):
                                st.session_state.quiz_score += 1
                            if i < len(st.session_state.mcqs) - 1:
                                st.session_state.quiz_index += 1

                        if cols[1].button("Skip"):
                            if i < len(st.session_state.mcqs) - 1:
                                st.session_state.quiz_index += 1

                        if cols[2].button("Previous"):
                            if i > 0:
                                st.session_state.quiz_index -= 1

                    if st.button("✅ Submit Quiz"):
                        total = len(st.session_state.mcqs)
                        st.success(f"Quiz Complete! Your Score: {st.session_state.quiz_score} / {total}")
                        for idx, q in enumerate(st.session_state.mcqs):
                            st.markdown(f"**Q{idx+1}. {q['question']}**")
                            st.markdown("\n".join(q['options']))
                            st.markdown(f"✅ Correct: {q['answer']}")
                            st.markdown(f"🧍 Your Answer: {st.session_state.user_answers[idx] if st.session_state.user_answers[idx] else 'Skipped'}")
                        st.session_state.mcqs = []

            elif mcq_mode == "📖 Q&A with Answers":
                if st.button("Generate MCQs"):
                    mcq_text = generate_mcqs(st.session_state.text, difficulty=level)
                    st.session_state.mcqs = parse_mcqs(mcq_text)
                for q in st.session_state.mcqs:
                    st.markdown(f"**{q['question']}**")
                    st.markdown("\n".join(q['options']))
                    st.markdown(f"✅ Answer: {q['answer']}")
else:
    st.info("👈 Please enter your Groq API key to get started.")
