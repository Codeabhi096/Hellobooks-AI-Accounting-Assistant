import os
import pickle
import time

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_PATH    = "vector_store/faiss.index"
DOCS_PATH     = "vector_store/documents.pkl"
DATA_DIR      = "data"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

SAMPLE_QUESTIONS = [
    "What is bookkeeping?",
    "How does a balance sheet work?",
    "What is the difference between revenue and profit?",
]

st.set_page_config(page_title="Hellobooks AI", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.stApp{background:#0d0f14;background-image:radial-gradient(ellipse at 15% 15%,rgba(99,179,112,.07) 0%,transparent 55%),radial-gradient(ellipse at 85% 85%,rgba(56,139,253,.05) 0%,transparent 55%);}
section[data-testid="stSidebar"]{background:#111318 !important;border-right:1px solid #1e2330;}
section[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
.hero-wrap{text-align:center;padding:2rem 1rem 1rem;}
.hero-title{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:700;background:linear-gradient(135deg,#63b370 0%,#3ecf8e 45%,#388bfd 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-.5px;margin:0 0 .3rem;}
.hero-sub{color:#8b949e;font-size:clamp(.82rem,2vw,1rem);font-weight:300;}
.card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;}
.card-answer{background:linear-gradient(135deg,#0d1f12 0%,#0d1520 100%);border:1px solid #238636;border-radius:12px;padding:1.5rem;margin-top:1rem;animation:fadeUp .35s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.answer-label{color:#3ecf8e;font-size:.7rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.7rem;}
.answer-text{color:#e6edf3;font-size:1rem;line-height:1.8;font-weight:300;}
.sources-row{margin-top:1rem;display:flex;flex-wrap:wrap;gap:.4rem;}
.source-chip{background:#1f2937;border:1px solid #374151;border-radius:20px;padding:3px 10px;font-size:.7rem;color:#9ca3af;font-family:'JetBrains Mono',monospace;}
.empty-state{text-align:center;padding:3.5rem 1rem;border:1px dashed #21262d;border-radius:16px;margin-top:1.5rem;}
.empty-icon{font-size:2.4rem;margin-bottom:.8rem;}
.empty-title{font-size:1rem;font-weight:600;color:#c9d1d9;margin-bottom:.4rem;}
.empty-sub{font-size:.85rem;color:#6e7681;line-height:1.6;}
.stTextArea textarea{background:#161b22 !important;border:1px solid #30363d !important;border-radius:10px !important;color:#e6edf3 !important;font-family:'Sora',sans-serif !important;font-size:.95rem !important;resize:none !important;}
.stTextArea textarea:focus{border-color:#238636 !important;box-shadow:0 0 0 3px rgba(35,134,54,.12) !important;}
.stButton>button{background:linear-gradient(135deg,#238636,#2ea043) !important;color:white !important;border:none !important;border-radius:8px !important;font-family:'Sora',sans-serif !important;font-weight:600 !important;font-size:.9rem !important;transition:all .2s !important;}
.stButton>button:hover{background:linear-gradient(135deg,#2ea043,#3ecf8e) !important;transform:translateY(-1px) !important;}
.stat-box{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:.8rem;text-align:center;}
.stat-num{font-size:1.6rem;font-weight:700;color:#3ecf8e;}
.stat-label{font-size:.68rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;}
.history-item{background:#161b22;border-left:3px solid #238636;border-radius:0 8px 8px 0;padding:.65rem .9rem;margin-bottom:.5rem;}
.history-q{font-size:.8rem;color:#c9d1d9;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.history-time{font-size:.65rem;color:#484f58;font-family:'JetBrains Mono',monospace;}
.score-bar-bg{background:#21262d;border-radius:4px;height:5px;overflow:hidden;margin-top:4px;}
.score-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#238636,#3ecf8e);}
hr{border-color:#21262d !important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:#0d0f14;}::-webkit-scrollbar-thumb{background:#21262d;border-radius:3px;}
@media(max-width:768px){.hero-wrap{padding:1.2rem .5rem .8rem;}.card-answer{padding:1.1rem;}}
</style>
""", unsafe_allow_html=True)


class Document:
    def __init__(self, content, source, chunk_index):
        self.content = content; self.source = source; self.chunk_index = chunk_index


def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start+CHUNK_SIZE].strip()
        if chunk: chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_and_index(data_dir, embed_model):
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".md"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                text = f.read().strip()
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(Document(chunk, fname, i))
    embs = embed_model.encode([d.content for d in docs], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, docs


@st.cache_resource(show_spinner=False)
def load_rag():
    model = SentenceTransformer(EMBED_MODEL)
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        idx = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f: docs = pickle.load(f)
        return model, idx, docs
    if not os.path.exists(DATA_DIR): return None, None, None
    idx, docs = load_and_index(DATA_DIR, model)
    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(idx, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f: pickle.dump(docs, f)
    return model, idx, docs


def ask_rag(question, embed_model, index, documents, api_key, model, top_k=3):
    q_vec = embed_model.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_vec, top_k)
    results = [(documents[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]
    context = "\n\n---\n\n".join([r[0].content for r in results])
    prompt = f"""You are a senior accounting expert and educator at Hellobooks.

Your job is to give DETAILED, COMPREHENSIVE answers to accounting questions.

Use the context below as your PRIMARY source. You may also use your own accounting knowledge to EXPAND and ENRICH the answer beyond what is in the context — but never contradict the context.

Structure your answer like this:
1. Start with a clear 2-3 sentence definition
2. Explain HOW it works with step-by-step detail
3. Give a real-world practical EXAMPLE with numbers
4. Mention WHY it matters for small business owners
5. Add any important tips or common mistakes to avoid

Write in clear, simple English. Use paragraphs. Minimum 250 words. Be thorough and educational.

Context:
{context}

Question: {question}
Answer:"""
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a friendly accounting assistant for small business owners."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7, max_tokens=1500,
    )
    return resp.choices[0].message.content.strip(), [r[0].source for r in results], [r[1] for r in results]


for k, v in [("history", []), ("q_input", ""), ("last", None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    api_key = (
        st.secrets.get("OPENROUTER_API_KEY", None)
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    clr = "#238636" if api_key else "#f85149"
    bg  = "#0d1f12"  if api_key else "#2d1212"
    txt = "Connected" if api_key else "Set OPENROUTER_API_KEY in .env"
    st.markdown(f"""<div style='background:{bg};border:1px solid {clr};border-radius:8px;
    padding:.5rem .9rem;font-size:.75rem;color:{clr};margin-bottom:.5rem'>● {txt}</div>""", unsafe_allow_html=True)

    model_choice = st.selectbox("Model", [
        "mistralai/mistral-7b-instruct",
        "google/gemma-3-12b-it:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "deepseek/deepseek-r1:free",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o-mini",
    ], index=0)

    top_k = st.slider("Chunks to retrieve", 1, 5, 3)
    st.markdown("---")

    embed_model, faiss_index, documents = load_rag()

    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class="stat-box"><div class="stat-num">{faiss_index.ntotal if faiss_index else 0}</div><div class="stat-label">Chunks</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="stat-box"><div class="stat-num">{len(st.session_state.history)}</div><div class="stat-label">Asked</div></div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕘 Recent")
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"""<div class="history-item">
                <div class="history-q">{item['q'][:52]}{'...' if len(item['q'])>52 else ''}</div>
                <div class="history-time">{item['t']}</div></div>""", unsafe_allow_html=True)
        if st.button("Clear history"):
            st.session_state.history = []; st.session_state.last = None; st.rerun()

# ── Main ──
st.markdown("""<div class="hero-wrap">
    <div class="hero-title">📚 Hellobooks AI</div>
    <div class="hero-sub">Your intelligent accounting assistant — powered by RAG</div>
</div>""", unsafe_allow_html=True)
st.markdown("---")

question = st.text_area("q", label_visibility="collapsed",
    placeholder="Ask an accounting question — e.g. What is cash flow?",
    value=st.session_state.q_input, height=90, key="q_box")

b1, b2 = st.columns([5, 1])
with b1: ask_clicked = st.button("Get Answer →", use_container_width=True)
with b2:
    if st.button("Clear", use_container_width=True):
        st.session_state.q_input = ""; st.session_state.last = None; st.rerun()

st.markdown("<p style='color:#6e7681;font-size:.78rem;margin:.6rem 0 .3rem'>Try an example:</p>", unsafe_allow_html=True)
cols = st.columns(3)
for i, sq in enumerate(SAMPLE_QUESTIONS):
    with cols[i]:
        if st.button(sq, key=f"sq_{i}", use_container_width=True):
            st.session_state.q_input = sq; st.rerun()

st.markdown("---")

if ask_clicked and question.strip():
    if not api_key:
        st.error("API key not found. Set OPENROUTER_API_KEY in your .env or Streamlit secrets.")
    elif embed_model is None:
        st.error("Knowledge base not found. Run `python main.py` first.")
    else:
        with st.spinner("Searching knowledge base..."):
            try:
                answer, sources, scores = ask_rag(question.strip(), embed_model, faiss_index, documents, api_key, model_choice, top_k)
                result = {"q": question.strip(), "a": answer, "sources": sources, "scores": scores, "t": time.strftime("%H:%M")}
                st.session_state.last = result
                st.session_state.history.append(result)
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.last:
    r = st.session_state.last
    uid = abs(hash(r['a'])) % 99999
    st.markdown(f"""
    <div class="card-answer">
        <div class="answer-label">Answer
            <span style="float:right;font-size:.68rem;color:#484f58;font-family:'JetBrains Mono',monospace">{r['t']}</span>
        </div>
        <div class="answer-text" id="ans-{uid}">{r['a']}</div>
        <div class="sources-row">
            {''.join(f'<span class="source-chip">📄 {s}</span>' for s in dict.fromkeys(r['sources']))}
        </div>
    </div>""", unsafe_allow_html=True)

    st.code(r['a'], language=None)

    with st.expander("View retrieved context"):
        for i, (src, score) in enumerate(zip(r['sources'], r['scores'])):
            st.markdown(f"**{src}** — similarity `{score:.3f}`")
            st.markdown(f"""<div class="score-bar-bg"><div class="score-bar-fill" style="width:{int(score*100)}%"></div></div>""", unsafe_allow_html=True)
            matching = [d for d in documents if d.source == src]
            if matching: st.caption(matching[i % len(matching)].content[:280] + "...")
            st.markdown("")

elif not ask_clicked:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📊</div>
        <div class="empty-title">Your answers start here</div>
        <div class="empty-sub">
            Ask about bookkeeping, invoices, cash flow, balance sheets,<br>
            taxes, revenue, profit — and get clear, practical answers instantly.
        </div>
    </div>""", unsafe_allow_html=True)