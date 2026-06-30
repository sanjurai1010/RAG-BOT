"""
app.py  ←  ENTRY POINT
Run: python -m streamlit run app.py
"""
import os
import streamlit as st
from dotenv import load_dotenv
from frontend.styles import load_css
from frontend.sidebar import render_sidebar
from frontend.chat import render_chat

load_dotenv()
st.session_state["temp_api_key"] = os.getenv("GROQ_API_KEY", "")

st.set_page_config(
    page_title="DocMind – RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "messages":     [],
    "vector_store": None,
    "chain":        None,
    "doc_count":    0,
    "chunk_count":  0,
    "file_names":   [],
    "is_csv":       False,
    "has_tables":   False,
    "dataframe":    None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

load_css()
render_sidebar()
render_chat()