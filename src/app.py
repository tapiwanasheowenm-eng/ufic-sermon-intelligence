import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="UFIC Sermon Intelligence System",
    page_icon="📖",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL CSS (Mobile Safe, Professional)
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f4f0fa;
}

/* Force all labels black */
label, .stSelectbox label, .stTextInput label {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Force dropdown + input text black */
div[data-baseweb="select"] * {
    color: #000000 !important;
}

input {
    color: #000000 !important;
}

/* Headers */
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #5e2b97;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #444444;
    margin-bottom: 25px;
}

/* Result Cards */
.result-card {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 18px;
    border-left: 6px solid #c9a227;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    color: #000000 !important;
}

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
    color: #555;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="main-title">UFIC Sermon Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Sermon Navigation & Structured Study Assistant</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL + INDEX
# ---------------------------------------------------
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

# ---------------------------------------------------
# NAVIGATION
# ---------------------------------------------------
page = st.radio(
    "Navigation",
    ["🔎 Search", "📚 Sermon Library", "📖 Scripture Explorer"],
    horizontal=True
)

# ---------------------------------------------------
# SEARCH PAGE
# ---------------------------------------------------
if page == "🔎 Search":

    query = st.text_input("Search sermon content")

    col1, col2, col3 = st.columns(3)

    with col1:
        year_filter = st.selectbox("Year", ["All"] + sorted(list(set(m["year"] for m in metadata))))

    with col2:
        month_filter = st.selectbox("Month", ["All"] + sorted(list(set(m["month"].title() for m in metadata))))

    with col3:
        results_per_page = st.selectbox("Results", [5, 10, 20], index=0)

    def highlight_text(text, keyword):
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub(lambda m: f"<span class='highlight'>{m.group()}</span>", text)

    if query:

        query_embedding = model.encode([query]).astype("float32")
        D, I = index.search(query_embedding, 100)

        filtered_results = []

        for idx in I[0]:
            result = metadata[idx]

            if year_filter != "All" and result["year"] != year_filter:
                continue

            if month_filter != "All" and result["month"].title() != month_filter:
                continue

            text_lower = result["text"].lower()
            query_lower = query.lower()

            keyword_match = query_lower in text_lower
            semantic_score = D[0][list(I[0]).index(idx)]

            if keyword_match or semantic_score < 1.15:
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

# ---------------------------------------------------
# SERMON LIBRARY
# ---------------------------------------------------
elif page == "📚 Sermon Library":

    st.markdown("### Sermon Archive")

    grouped = defaultdict(list)

    for item in metadata:
        grouped[item["title"]].append(item)

    for title, items in grouped.items():
        sermon = items[0]
        youtube_url = f"https://www.youtube.com/watch?v={sermon['youtube_id']}"

        st.markdown(f"""
        <div class="result-card">
            <b>{title}</b><br>
            Date: {sermon['date']} | Event: {sermon['event']}<br><br>
            ▶ <a href="{youtube_url}" target="_blank">Watch Full Sermon</a>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# SCRIPTURE EXPLORER
# ---------------------------------------------------
elif page == "📖 Scripture Explorer":

    st.markdown("### Scripture Explorer")

    scripture_map = defaultdict(list)

    for item in metadata:
        if "scriptures" in item:
            for s in item["scriptures"]:
                scripture_map[s].append(item)

    selected_scripture = st.selectbox(
        "Select Scripture",
        ["Select"] + sorted(scripture_map.keys())
    )

    if selected_scripture != "Select":
        sermons = scripture_map[selected_scripture]

        for sermon in sermons:
            youtube_url = f"https://www.youtube.com/watch?v={sermon['youtube_id']}&t={int(sermon['start'])}s"

            st.markdown(f"""
            <div class="result-card">
                <b>{sermon['title']}</b><br>
                Date: {sermon['date']} | Event: {sermon['event']}<br><br>
                ▶ <a href="{youtube_url}" target="_blank">Watch Reference</a>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------
# HIDDEN ADMIN ACCESS
# ---------------------------------------------------
query_params = st.query_params

if query_params.get("admin") == "true":
    st.markdown("## 🔐 Admin Dashboard")
    st.write(f"Total Sermon Chunks Indexed: {len(metadata)}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class="footer">
UFIC Sermon Intelligence v1.0 — Internal Prototype<br>
This tool assists with sermon navigation and structured study.
It does not replace spiritual discernment or personal revelation.
</div>
""", unsafe_allow_html=True)