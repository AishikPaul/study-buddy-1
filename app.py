# study_buddy_app.py — Rebuilt & Improved

import streamlit as st
import os
import re
import faiss
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────── Page Config ───────────────────────────

st.set_page_config(
    page_title="Study Buddy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Flashcard */
.flashcard {
    background: linear-gradient(135deg, rgba(100,80,200,0.15), rgba(60,120,220,0.1));
    border: 1px solid rgba(120,100,220,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.flashcard .fc-q { font-weight: 600; font-size: 1rem; color: #c0b4f0; margin-bottom: 0.5rem; }
.flashcard .fc-a { font-size: 0.95rem; color: #a8d4f0; }

/* MCQ */
.mcq-block {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

/* Chat bubble */
.chat-user {
    background: rgba(100,80,200,0.2);
    border-left: 3px solid #7b6ee0;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}
.chat-bot {
    background: rgba(60,120,220,0.15);
    border-left: 3px solid #4a9edd;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
}

/* Correct / wrong in quiz review */
.correct { color: #6ee06e; font-weight: 600; }
.wrong { color: #e06e6e; font-weight: 600; }

/* Score badge */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6b4de6, #4a9edd);
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
}

h1, h2, h3 { font-family: 'Sora', sans-serif; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6b4de6, #4a9edd);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Radio / Select */
.stRadio label, .stSelectbox label { color: #c0b8d8; }

/* Text input */
.stTextInput input {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    color: #e8e8f0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Session State Init ───────────────────────────

def init_state():
    defaults = {
        "text": None,
        "small_chunks": [],
        "big_chunks": [],
        "small_index": None,
        "big_index": None,
        "mcqs": [],
        "quiz_index": 0,
        "quiz_score": 0,
        "user_answers": [],
        "quiz_submitted": False,
        "chat_history": [],       # list of {"role": "user"/"bot", "content": str}
        "summary_cache": None,
        "flashcard_cache": None,
        "doc_name": None,
        "embed_model": None,
        "llm": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────── Model Loading ───────────────────────────

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# ─────────────────────────── Utility Functions ───────────────────────────

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(pages)
    elif name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif name.endswith(".txt"):
        return file.read().decode("utf-8")
    return None

def smart_truncate(text, max_chars=12000):
    """Truncate text smartly to avoid LLM token overflow."""
    if len(text) <= max_chars:
        return text
    # Take a representative sample: beginning, middle, end
    chunk = max_chars // 3
    return text[:chunk] + "\n\n...[middle content]...\n\n" + text[len(text)//2 - chunk//2 : len(text)//2 + chunk//2] + "\n\n...[continuing]...\n\n" + text[-chunk:]

def chunk_text(text):
    sections = split_into_sections(text)
    return get_dual_chunks_for_sections(sections)

def split_into_sections(text):
    sections, current = [], []
    for line in text.split('\n'):
        if re.match(r'^([A-Z\s]{5,}|.{3,60}:)$', line.strip()):
            if current:
                sections.append('\n'.join(current).strip())
                current = []
        current.append(line)
    if current:
        sections.append('\n'.join(current).strip())
    return sections if sections else [text]

def get_dual_chunks_for_sections(sections):
    all_small, all_big = [], []
    for section in sections:
        length = len(section)
        if length < 1000:
            s_size, s_ov, b_size, b_ov = 300, 30, 700, 100
        elif length < 3000:
            s_size, s_ov, b_size, b_ov = 500, 80, 1200, 150
        else:
            s_size, s_ov, b_size, b_ov = 600, 100, 1500, 200
        all_small.extend(RecursiveCharacterTextSplitter(chunk_size=s_size, chunk_overlap=s_ov).split_text(section))
        all_big.extend(RecursiveCharacterTextSplitter(chunk_size=b_size, chunk_overlap=b_ov).split_text(section))
    return all_small, all_big

def build_index(chunks):
    vecs = embed_model.encode(chunks, show_progress_bar=False)
    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)
    return idx

def retrieve_context(question, small_chunks, big_chunks, small_index, big_index, top_k=3):
    q_vec = embed_model.encode([question])
    _, si = small_index.search(np.array(q_vec), top_k)
    _, bi = big_index.search(np.array(q_vec), top_k)
    small_ctx = "\n".join([small_chunks[i] for i in si[0] if i < len(small_chunks)])
    big_ctx   = "\n".join([big_chunks[i]   for i in bi[0]   if i < len(big_chunks)])
    return small_ctx + "\n\n" + big_ctx

def llm_invoke(prompt, system=None):
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    return st.session_state.llm.invoke(msgs).content

# ─────────────────────────── Feature Functions ───────────────────────────

def ask_question(question):
    context = retrieve_context(
        question,
        st.session_state.small_chunks,
        st.session_state.big_chunks,
        st.session_state.small_index,
        st.session_state.big_index,
    )
    system = (
        "You are a precise study assistant. Answer ONLY from the given context. "
        "If the answer is not found, say: 'This doesn't appear to be covered in the document.' "
        "Be clear, structured, and concise."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return llm_invoke(prompt, system=system)

def generate_summary(text):
    truncated = smart_truncate(text, 14000)
    system = "You are a study assistant. Create a well-structured summary with key points."
    prompt = (
        "Summarize the following study material clearly. "
        "Use bullet points for key concepts, then 2-3 sentences of conclusion.\n\n"
        f"{truncated}"
    )
    return llm_invoke(prompt, system=system)

def generate_flashcards(text, n=8):
    truncated = smart_truncate(text, 10000)
    system = "You are a study assistant creating effective flashcards."
    prompt = (
        f"Create exactly {n} flashcard pairs from this content. "
        "Format each as:\nQ: [question]\nA: [answer]\n\n"
        "Make questions test understanding, not just recall.\n\n"
        f"Content:\n{truncated}"
    )
    return llm_invoke(prompt, system=system)

def parse_flashcards(raw):
    """Parse Q:/A: formatted flashcards into list of dicts."""
    cards = []
    blocks = re.split(r'\n(?=Q:)', raw.strip())
    for block in blocks:
        q_match = re.search(r'Q:\s*(.+?)(?=\nA:)', block, re.DOTALL)
        a_match = re.search(r'A:\s*(.+)', block, re.DOTALL)
        if q_match and a_match:
            cards.append({
                "q": q_match.group(1).strip(),
                "a": a_match.group(1).strip()
            })
    return cards

def generate_mcqs(text, difficulty="medium", n=5):
    truncated = smart_truncate(text, 10000)
    system = "You are a quiz creator. Output ONLY the questions in the exact format specified, nothing else."
    prompt = f"""Generate exactly {n} multiple-choice questions at {difficulty} difficulty.

STRICT FORMAT (no deviations):
Q1. [question]
A. [option]
B. [option]
C. [option]
D. [option]
Answer: [letter only, e.g. B]

Q2. [question]
...

Content:
{truncated}"""
    return llm_invoke(prompt, system=system)

def parse_mcqs(raw):
    """Robust MCQ parser that handles LLM formatting variations."""
    questions = []
    # Split on question markers Q1, Q2, ... or numbered lines
    blocks = re.split(r'\n(?=Q\d+\.)', raw.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if not lines:
            continue
        # Question line
        q_line = lines[0]
        q_text = re.sub(r'^Q\d+\.\s*', '', q_line).strip()
        # Options
        options = []
        answer_letter = None
        for line in lines[1:]:
            opt_match = re.match(r'^([A-D])[.)]\s+(.+)', line)
            ans_match = re.match(r'^Answer:\s*([A-D])', line, re.IGNORECASE)
            if opt_match:
                options.append(f"{opt_match.group(1)}. {opt_match.group(2).strip()}")
            elif ans_match:
                answer_letter = ans_match.group(1).upper()
        if q_text and len(options) == 4 and answer_letter:
            questions.append({
                "question": q_text,
                "options": options,
                "answer": answer_letter,
            })
    return questions

# ─────────────────────────── Sidebar ───────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Study Buddy")
    st.markdown("---")

    api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    if api_key and st.session_state.llm is None:
        st.session_state.llm = ChatGroq(
            temperature=0.5,
            model_name="llama3-70b-8192",   # upgraded from 8b → 70b for quality
            api_key=api_key,
        )

    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Study Material", type=["pdf", "txt", "docx"])

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("📖 Processing document…"):
            text = extract_text(uploaded_file)
            if text:
                small_chunks, big_chunks = chunk_text(text)
                st.session_state.text = text
                st.session_state.small_chunks = small_chunks
                st.session_state.big_chunks = big_chunks
                st.session_state.small_index = build_index(small_chunks)
                st.session_state.big_index = build_index(big_chunks)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.summary_cache = None
                st.session_state.flashcard_cache = None
                st.session_state.chat_history = []
                st.session_state.mcqs = []
            else:
                st.error("Could not extract text from this file.")

    if st.session_state.doc_name:
        st.success(f"✅ {st.session_state.doc_name}")
        words = len(st.session_state.text.split()) if st.session_state.text else 0
        st.caption(f"~{words:,} words · {len(st.session_state.small_chunks)} chunks")

    st.markdown("---")
    if not api_key:
        st.info("Enter your API key to start.")
    elif not st.session_state.text:
        st.info("Upload a document to begin.")
    else:
        menu = st.radio(
            "Feature",
            ["💬 Ask Questions", "📋 Summary", "🃏 Flashcards", "📝 MCQ Quiz"],
            label_visibility="collapsed",
        )

# ─────────────────────────── Main Area ───────────────────────────

if not api_key:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <h1 style="font-size: 2.5rem; background: linear-gradient(135deg, #7b6ee0, #4a9edd); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🧠 Study Buddy
        </h1>
        <p style="color: #888; font-size: 1.1rem; margin-top: 1rem;">
            Your AI-powered study companion.<br>
            Enter your Groq API key in the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.text:
    st.markdown("""
    <div style="text-align:center; padding: 3rem;">
        <h2 style="color: #888;">Upload a document to get started 📂</h2>
        <p style="color: #666;">Supports PDF, DOCX, and TXT files</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════ 💬 ASK QUESTIONS ═══════════════════════
if menu == "💬 Ask Questions":
    st.markdown("## 💬 Ask Questions")
    st.caption("Ask anything about your document — answers are grounded strictly in the content.")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("Your question", placeholder="What is the main topic of this document?", label_visibility="collapsed")
    with col2:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            answer = ask_question(question)
        st.session_state.chat_history.append({"role": "bot", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════ 📋 SUMMARY ═══════════════════════
elif menu == "📋 Summary":
    st.markdown("## 📋 Document Summary")

    if st.session_state.summary_cache:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(st.session_state.summary_cache)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🔄 Regenerate"):
            st.session_state.summary_cache = None
            st.rerun()
    else:
        if st.button("✨ Generate Summary"):
            with st.spinner("Summarizing your document…"):
                summary = generate_summary(st.session_state.text)
            st.session_state.summary_cache = summary
            st.rerun()

# ═══════════════════════ 🃏 FLASHCARDS ═══════════════════════
elif menu == "🃏 Flashcards":
    st.markdown("## 🃏 Flashcards")

    col1, col2 = st.columns([3, 1])
    with col1:
        n_cards = st.slider("Number of flashcards", 4, 15, 8)
    with col2:
        gen_btn = st.button("✨ Generate", use_container_width=True)

    if gen_btn:
        with st.spinner("Creating flashcards…"):
            raw = generate_flashcards(st.session_state.text, n=n_cards)
            cards = parse_flashcards(raw)
        if cards:
            st.session_state.flashcard_cache = cards
        else:
            st.warning("Couldn't parse flashcards. Try regenerating.")

    if st.session_state.flashcard_cache:
        cards = st.session_state.flashcard_cache
        st.caption(f"{len(cards)} flashcards generated")
        for i, card in enumerate(cards, 1):
            with st.expander(f"Card {i} — {card['q'][:60]}{'…' if len(card['q']) > 60 else ''}"):
                st.markdown(f'<div class="flashcard"><div class="fc-q">❓ {card["q"]}</div><div class="fc-a">💡 {card["a"]}</div></div>', unsafe_allow_html=True)

# ═══════════════════════ 📝 MCQ QUIZ ═══════════════════════
elif menu == "📝 MCQ Quiz":
    st.markdown("## 📝 MCQ Quiz")

    mcq_mode = st.radio("Mode", ["📖 Study Mode (with answers)", "🧪 Practice Quiz"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
    with col2:
        n_questions = st.slider("Number of questions", 3, 10, 5)

    # ── Study Mode ──
    if mcq_mode == "📖 Study Mode (with answers)":
        if st.button("✨ Generate MCQs"):
            with st.spinner("Generating questions…"):
                raw = generate_mcqs(st.session_state.text, difficulty, n_questions)
                st.session_state.mcqs = parse_mcqs(raw)
                st.session_state.quiz_submitted = False

        if st.session_state.mcqs:
            for i, q in enumerate(st.session_state.mcqs, 1):
                st.markdown(f'<div class="mcq-block">', unsafe_allow_html=True)
                st.markdown(f"**Q{i}. {q['question']}**")
                for opt in q['options']:
                    letter = opt[0]
                    marker = "✅ " if letter == q['answer'] else "　 "
                    st.markdown(f"{marker}{opt}")
                st.markdown(f"**Answer: {q['answer']}**")
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Practice Quiz ──
    else:
        if st.button("🚀 Start New Quiz"):
            with st.spinner("Generating quiz…"):
                raw = generate_mcqs(st.session_state.text, difficulty, n_questions)
                parsed = parse_mcqs(raw)
            if parsed:
                st.session_state.mcqs = parsed
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.user_answers = [None] * len(parsed)
                st.session_state.quiz_submitted = False
            else:
                st.error("Failed to generate questions. Try again.")

        if st.session_state.mcqs and not st.session_state.quiz_submitted:
            mcqs = st.session_state.mcqs
            total = len(mcqs)

            # Progress bar
            answered = sum(1 for a in st.session_state.user_answers if a is not None)
            st.progress(answered / total, text=f"Answered {answered} / {total}")

            # Question palette
            palette_cols = st.columns(min(total, 10))
            for idx in range(total):
                label = "✅" if st.session_state.user_answers[idx] else str(idx + 1)
                if palette_cols[idx % 10].button(label, key=f"pal_{idx}"):
                    st.session_state.quiz_index = idx
                    st.rerun()

            st.markdown("---")
            i = st.session_state.quiz_index
            q = mcqs[i]

            st.markdown(f"### Q{i+1} of {total}")
            st.markdown(f"**{q['question']}**")

            # Pre-select previously saved answer
            prev = st.session_state.user_answers[i]
            prev_idx = next((j for j, o in enumerate(q['options']) if o[0] == prev), 0) if prev else 0

            choice = st.radio(
                "Choose:",
                q['options'],
                index=prev_idx,
                key=f"quiz_{i}",
                label_visibility="collapsed",
            )

            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

            if nav_col1.button("← Prev") and i > 0:
                st.session_state.user_answers[i] = choice[0]
                st.session_state.quiz_index -= 1
                st.rerun()

            if nav_col2.button("Save & Next →"):
                st.session_state.user_answers[i] = choice[0]
                if i < total - 1:
                    st.session_state.quiz_index += 1
                st.rerun()

            if nav_col3.button("Skip →") and i < total - 1:
                st.session_state.quiz_index += 1
                st.rerun()

            if nav_col4.button("✅ Submit Quiz"):
                # Save current answer before submitting
                st.session_state.user_answers[i] = choice[0]
                # Calculate score (count correct answers, not cumulative)
                score = sum(
                    1 for j, mcq in enumerate(mcqs)
                    if st.session_state.user_answers[j] == mcq['answer']
                )
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                st.rerun()

        # ── Quiz Results ──
        if st.session_state.quiz_submitted and st.session_state.mcqs:
            mcqs = st.session_state.mcqs
            total = len(mcqs)
            score = st.session_state.quiz_score
            pct = int(score / total * 100)

            if pct >= 80:
                grade, emoji = "Excellent!", "🏆"
            elif pct >= 60:
                grade, emoji = "Good job!", "👍"
            else:
                grade, emoji = "Keep studying!", "📚"

            st.markdown(f"""
            <div style="text-align:center; padding: 1.5rem;">
                <div style="font-size: 3rem;">{emoji}</div>
                <div class="score-badge">{score} / {total} — {pct}%</div>
                <p style="color:#888; margin-top:0.5rem;">{grade}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Review")
            for idx, q in enumerate(mcqs):
                user_ans = st.session_state.user_answers[idx]
                correct_ans = q['answer']
                is_correct = user_ans == correct_ans
                with st.expander(f"Q{idx+1}. {q['question'][:70]}{'…' if len(q['question'])>70 else ''}  {'✅' if is_correct else '❌'}"):
                    for opt in q['options']:
                        letter = opt[0]
                        if letter == correct_ans:
                            st.markdown(f"<span class='correct'>✅ {opt}</span>", unsafe_allow_html=True)
                        elif letter == user_ans and not is_correct:
                            st.markdown(f"<span class='wrong'>❌ {opt} (your answer)</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"　 {opt}")
                    if user_ans is None:
                        st.caption("⚠️ Skipped")

            if st.button("🔄 Try Another Quiz"):
                st.session_state.mcqs = []
                st.session_state.quiz_submitted = False
                st.rerun()
