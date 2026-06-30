import os
import streamlit as st 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
import tempfile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0f14; color: #e8e4dc; }

[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e8e4dc;
    font-family: 'Syne', sans-serif;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f5c842 0%, #e8895a 60%, #c45c8a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1rem;
    color: #6b7280;
    letter-spacing: 0.04em;
    margin-bottom: 1.8rem;
}
.free-badge {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #34d399;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.msg-user { display: flex; justify-content: flex-end; margin-bottom: 0.8rem; }
.msg-assistant { display: flex; justify-content: flex-start; margin-bottom: 0.8rem; }
.bubble-user {
    background: linear-gradient(135deg, #f5c842, #e8895a);
    color: #0d0f14;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.93rem;
    font-weight: 500;
    line-height: 1.55;
    box-shadow: 0 4px 20px rgba(245,200,66,0.15);
}
.bubble-assistant {
    background: #1a1e2a;
    border: 1px solid #252936;
    color: #d4d0c8;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 0.93rem;
    line-height: 1.65;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.bubble-assistant strong { color: #f5c842; }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.badge-ready {
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34d399;
}
.badge-waiting {
    background: rgba(107, 114, 128, 0.12);
    border: 1px solid rgba(107, 114, 128, 0.3);
    color: #6b7280;
}
.file-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: #1a1e2a;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    border: 1px solid #252936;
    font-size: 0.85rem;
    color: #a0a8b8;
}
.info-box {
    background: rgba(245, 200, 66, 0.06);
    border: 1px solid rgba(245, 200, 66, 0.2);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    font-size: 0.82rem;
    color: #a0956b;
    line-height: 1.5;
    margin-bottom: 0.8rem;
}
.info-box a { color: #f5c842; text-decoration: none; }

[data-testid="stChatInput"] textarea {
    background: #1a1e2a !important;
    border: 1px solid #2e3347 !important;
    border-radius: 12px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.1) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #f5c842, #e8895a) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: #1f2330 !important; }
[data-testid="metric-container"] {
    background: #1a1e2a;
    border: 1px solid #252936;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] {
    color: #f5c842 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2e3347; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Groq model options ────────────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B  — Best quality",
    "llama-3.1-8b-instant":    "LLaMA 3.1 8B   — Fastest",
    "mixtral-8x7b-32768":      "Mixtral 8x7B   — Long context",
    "gemma2-9b-it":            "Gemma 2 9B     — Google model",
}

SUPPORTED = {
    "pdf":  ("📄", "PDF"),
    "txt":  ("📝", "Text"),
    "docx": ("📘", "Word"),
    "csv":  ("📊", "CSV"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_file(uploaded_file) -> list:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if ext == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext == "txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    elif ext == "docx":
        loader = Docx2txtLoader(tmp_path)
    elif ext == "csv":
        loader = CSVLoader(tmp_path)
    else:
        return []

    docs = loader.load()
    os.unlink(tmp_path)
    return docs


@st.cache_resource(show_spinner="Loading embedding model (first time only, ~1 min)…")
def get_embeddings():
    """Free lightweight FastEmbed — no PyTorch/CUDA required."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def build_chain(vector_store, groq_api_key, model, top_k, temperature):
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    llm = ChatGroq(model=model, temperature=temperature, api_key=groq_api_key)

    prompt = PromptTemplate(
        template="""You are a knowledgeable assistant. Answer ONLY using the provided document context.
If the answer is not in the context, say "I couldn't find that information in the uploaded documents."
Be concise but thorough. Cite relevant details from the documents.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vector_store": None,
    "chain": None,
    "doc_count": 0,
    "chunk_count": 0,
    "file_names": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown(
        '<div class="info-box">'
        "<strong>100% Free!</strong><br>"
        "Get your free Groq API key at<br>"
        '<a href="https://console.groq.com" target="_blank">console.groq.com</a>'
        " → Sign up → API Keys → Create Key<br>"
        "<em>No credit card needed.</em>"
        "</div>",
        unsafe_allow_html=True,
    )

    groq_api_key = st.session_state.get("temp_api_key", "")

    model_choice = st.selectbox(
        "LLM Model",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📂 Upload Documents")
    st.caption("Supported: PDF · TXT · DOCX · CSV")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=list(SUPPORTED.keys()),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    top_k       = st.slider("Retrieved chunks (k)", 2, 10, 4)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.2, 0.05)

    build_btn = st.button("🔨 Build Knowledge Base", use_container_width=True)

    if build_btn:
        if not groq_api_key:
            st.error("Please enter your Groq API key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Processing documents…"):
                try:
                    all_docs, names = [], []
                    for f in uploaded_files:
                        docs = load_file(f)
                        all_docs.extend(docs)
                        names.append(f.name)

                    vs, n_chunks = build_vector_store(all_docs)
                    chain = build_chain(vs, groq_api_key, model_choice, top_k, temperature)

                    st.session_state.vector_store = vs
                    st.session_state.chain        = chain
                    st.session_state.doc_count    = len(uploaded_files)
                    st.session_state.chunk_count  = n_chunks
                    st.session_state.file_names   = names
                    st.session_state.messages     = []
                    st.success("Knowledge base ready! ")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    if st.session_state.vector_store:
        st.markdown('<span class="status-badge badge-ready">● READY</span>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Docs",   st.session_state.doc_count)
        col2.metric("Chunks", st.session_state.chunk_count)

        st.markdown("**Indexed files:**")
        for name in st.session_state.file_names:
            ext  = name.rsplit(".", 1)[-1].lower()
            icon = SUPPORTED.get(ext, ("📎", ""))[0]
            st.markdown(f'<div class="file-item">{icon} {name}</div>', unsafe_allow_html=True)

        if st.button("🗑 Clear & Reset", use_container_width=True):
            for k in ["vector_store", "chain", "messages", "doc_count", "chunk_count", "file_names"]:
                st.session_state[k] = ([] if k in ["messages", "file_names"]
                                       else None if k in ["vector_store", "chain"]
                                       else 0)
            st.rerun()
    else:
        st.markdown('<span class="status-badge badge-waiting">○ NO DOCS LOADED</span>', unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">DocMind <span class="free-badge">FREE</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Powered by Groq · HuggingFace Embeddings · Ask anything about your files</div>',
    unsafe_allow_html=True,
)

# ── API Key input on main page ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
st.session_state["temp_api_key"] = os.getenv("GROQ_API_KEY", "")
if not st.session_state.vector_store:
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state["temp_api_key"]:
            st.success("✅ API Key loaded from .env file!")
        else:
            st.error("❌ No API key found in .env file!")
    with col2:
        st.markdown(
            '<div style="padding:0.6rem;background:rgba(245,200,66,0.08);border:1px solid rgba(245,200,66,0.2);'
            'border-radius:10px;font-size:0.8rem;color:#a0956b;margin-top:1.8rem;">'
            '🆓 Get free key at<br><a href="https://console.groq.com" style="color:#f5c842;">console.groq.com</a>'
            '</div>',
            unsafe_allow_html=True
        )
    st.divider()

with st.container():
    if not st.session_state.messages:
        if st.session_state.vector_store:
            st.markdown(
                '<div class="bubble-assistant" style="max-width:60%;margin:2rem auto;text-align:center;">'
                " Knowledge base is ready!<br>Ask me anything about your documents."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center;margin-top:4rem;">'
                "<div style='font-size:3rem;margin-bottom:1rem;'>📚</div>"
                "<div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#2e3347;'>"
                "Upload documents to get started</div>"
                "<div style='font-size:0.9rem;margin-top:0.5rem;color:#4a5060;'>"
                "Sidebar → add Groq key → upload files → Build Knowledge Base</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="bubble-user">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="msg-assistant"><div class="bubble-assistant">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )

if st.session_state.chain:
    if user_input := st.chat_input("Ask something about your documents…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking…"):
            try:
                answer = st.session_state.chain.invoke(user_input)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
else:
    st.chat_input("Upload documents and build the knowledge base first…", disabled=True)