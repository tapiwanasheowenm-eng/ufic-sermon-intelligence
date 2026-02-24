import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="UFIC Sermon Intelligence System",
    page_icon="📖",
    layout="wide"
)

# ---------------------------------------------------
# CSS FIX (FINAL STABLE VERSION)
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f4f0fa;
}

/* Force all text black */
html, body, [class*="css"]  {
    color: #000000 !important;
}

/* Radio navigation */
div[role="radiogroup"] label {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Inputs */
input, textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
    border: 2px solid #000000 !important;
    border-radius: 6px !important;
}

/* Enter key button color fix (mobile) */
button {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Select boxes */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 2px solid #000000 !important;
    border-radius: 6px !important;
}

div[data-baseweb="select"] * {
    color: #000000 !important;
}

/* Headers */
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #5e2b97 !important;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #444444 !important;
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
    color: #555 !important;
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
# SCRIPTURE EXPLORER (UPGRADED)
# ---------------------------------------------------
if page == "📖 Scripture Explorer":

    st.markdown("### Scripture Explorer")

    # Build book → chapter mapping
    scripture_map = defaultdict(lambda: defaultdict(list))

    for item in metadata:
        text = item["text"].lower()

        # Find book mentions
        book_match = re.findall(r"(genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|samuel|kings|chronicles|ezra|nehemiah|esther|job|psalms|proverbs|ecclesiastes|isaiah|jeremiah|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelation)", text)

        # Find chapter references like Genesis 1 or Genesis 1:3
        chapter_match = re.findall(r"(genesis|exodus|leviticus|numbers|deuteronomy|matthew|mark|luke|john)\s+(\d+)", text)

        for book in book_match:
            scripture_map[book.title()]["All"].append(item)

        for book, chapter in chapter_match:
            scripture_map[book.title()][chapter].append(item)

    if scripture_map:

        selected_book = st.selectbox("Select Book", sorted(scripture_map.keys()))

        chapter_options = sorted(scripture_map[selected_book].keys(), key=lambda x: (x!="All", x))
        selected_chapter = st.selectbox("Select Chapter", chapter_options)

        sermons = scripture_map[selected_book][selected_chapter]

        if sermons:
            for sermon in sermons:
                youtube_url = f"https://www.youtube.com/watch?v={sermon['youtube_id']}&t={int(sermon['start'])}s"

                st.markdown(f"""
                <div class="result-card">
                    <b>{sermon['title']}</b><br>
                    Date: {sermon['date']} | Event: {sermon['event']}<br><br>
                    ▶ <a href="{youtube_url}" target="_blank">Watch Reference</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No sermons found for that chapter.")

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