import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from datetime import datetime
from collections import defaultdict

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="UFIC Sermon Intelligence System",
    page_icon="📖",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS (Mobile Safe)
# ---------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f3f0fa;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: #5e2b97;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #6c6c6c;
    margin-bottom: 30px;
}

/* Result Cards */
.result-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border-left: 6px solid #b38f00;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    color: #000000 !important;
}

/* Force text color black inside cards */
.result-card * {
    color: #000000 !important;
}

/* Highlight */
.highlight {
    background-color: #fff3a3;
    padding: 2px 4px;
    border-radius: 4px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #888;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="main-title">UFIC Sermon Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Sermon Navigation & Structured Study Assistant</div>', unsafe_allow_html=True)

# ---------------------------
# LOAD INDEX
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    index = faiss.read_index("data/index/sermon_index.faiss")
    with open("data/index/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

model = load_model()
index, metadata = load_index()

# ---------------------------
# SEARCH INPUT
# ---------------------------
query = st.text_input("Search sermon content")

col1, col2, col3 = st.columns(3)

with col1:
    year_filter = st.selectbox("Year", ["All"] + sorted(list(set(m["year"] for m in metadata))))

with col2:
    month_filter = st.selectbox("Month", ["All"] + sorted(list(set(m["month"].title() for m in metadata))))

with col3:
    results_per_page = st.selectbox("Results", [5, 10, 20], index=0)

# ---------------------------
# HIGHLIGHT FUNCTION
# ---------------------------
def highlight_text(text, keyword):
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"<span class='highlight'>{m.group()}</span>", text)

# ---------------------------
# SEARCH EXECUTION
# ---------------------------
if query:

    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, 100)

    filtered_results = []

    for idx in I[0]:
        result = metadata[idx]

        # Year filter
        if year_filter != "All" and result["year"] != year_filter:
            continue

        # Month filter
        if month_filter != "All" and result["month"].title() != month_filter:
            continue

        text_lower = result["text"].lower()
        query_lower = query.lower()

        # Hybrid check
        keyword_match = query_lower in text_lower
        semantic_score = D[0][list(I[0]).index(idx)]

        if keyword_match or semantic_score < 1.2:
            filtered_results.append((result, semantic_score, keyword_match))

    if not filtered_results:
        st.warning("No relevant results found.")
    else:
        filtered_results = sorted(filtered_results, key=lambda x: (not x[2], x[1]))

        st.markdown(f"### Showing {min(len(filtered_results), results_per_page)} relevant sermons")

        for result, score, keyword_match in filtered_results[:results_per_page]:

            highlighted_text = highlight_text(result["text"], query) if keyword_match else result["text"]

            youtube_url = f"https://www.youtube.com/watch?v={result['youtube_id']}&t={int(result['start'])}s"

            st.markdown(f"""
            <div class="result-card">
                <b>{result['title']}</b><br>
                Date: {result['date']} | Event: {result['event']}<br><br>
                ▶ <a href="{youtube_url}" target="_blank">Watch on YouTube (Start at {int(result['start'])}s)</a><br><br>
                📍 Timestamp: {round(result['start'],2)} sec<br><br>
                {highlighted_text}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
<div class="footer">
UFIC Sermon Intelligence v1.0 — Internal Prototype<br>
This tool assists with sermon navigation and structured study.
It does not replace spiritual discernment or personal revelation.
</div>
""", unsafe_allow_html=True)