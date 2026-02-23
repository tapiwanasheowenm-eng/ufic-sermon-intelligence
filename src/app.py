import streamlit as st
import faiss
import json
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="UFIC Sermon Intelligence", layout="wide")

# =====================================================
# LIGHT UFIC BRAND STYLING
# =====================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #EDE3F7;
}
h1, h2, h3, h4 {
    color: #5A2D82 !important;
}
.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.watch-button {
    background-color: #D4AF37;
    color: black !important;
    padding: 8px 14px;
    border-radius: 6px;
    text-decoration: none;
    font-weight: bold;
}
.footer-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    margin-top: 40px;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD RESOURCES
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    return faiss.read_index("data/index/sermon_index.faiss")

@st.cache_resource
def load_metadata():
    with open("data/index/metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
index = load_index()
metadata = load_metadata()

# =====================================================
# STOPWORDS
# =====================================================

STOPWORDS = set([
    "what","to","do","when","you","the","is","are","a","an","and",
    "of","in","on","for","with","how","your","be","i","me","my",
    "we","our","us","that","this","it","as","at","by","from"
])

def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def highlight(text, terms):
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(lambda x: f"<mark style='background-color:yellow'>{x.group()}</mark>", text)
    return text

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def detect_series(title):
    if "Prophetic Timeline" in title:
        return "Prophetic Timeline Series"
    if "Zaphnathpaaneah" in title:
        return "Zaphnathpaaneah Series"
    return "Standalone Sermons"

# =====================================================
# LIGHTWEIGHT THEME DETECTION
# =====================================================

THEME_KEYWORDS = {
    "End Times": ["rapture", "tribulation", "antichrist", "mark of the beast", "70 weeks"],
    "Faith": ["faith", "believe", "trust god"],
    "Kingdom": ["kingdom", "throne", "authority"],
    "Identity": ["identity", "sonship"],
    "Grace": ["grace", "favor"],
    "Judgment": ["judgment", "wrath"],
    "Covenant": ["covenant", "promise"],
    "Spiritual Warfare": ["warfare", "enemy", "devil", "demon"]
}

def detect_themes(text):
    text_lower = text.lower()
    detected = []
    for theme, keywords in THEME_KEYWORDS.items():
        for word in keywords:
            if word in text_lower:
                detected.append(theme)
                break
    return list(set(detected))

# =====================================================
# HEADER
# =====================================================

st.markdown("# UFIC Sermon Intelligence System")
st.markdown("### AI-Powered Sermon Navigation & Study Assistant")
st.markdown("""
<div style='text-align:center; font-size:14px; color:#5A2D82; margin-bottom:20px;'>
Structured navigation • Scripture intelligence • Theological analytics
</div>
""", unsafe_allow_html=True)

st.markdown("---")

page = st.sidebar.radio("Navigate", ["Search", "Sermon Library", "Scripture Explorer"])

# =====================================================
# SEARCH PAGE
# =====================================================

if page == "Search":

    years = sorted(list(set([m["year"] for m in metadata])))
    months = sorted(list(set([m["month"] for m in metadata])))

    all_scriptures = set()
    for m in metadata:
        for s in m.get("scriptures", []):
            all_scriptures.add(s)

    scripture_options = sorted(list(all_scriptures))

    col1, col2, col3, col4, col5 = st.columns([4,1,1,1,1])

    with col1:
        query = st.text_input("Search sermon content")

    with col2:
        year_filter = st.selectbox("Year", ["All"] + years)

    with col3:
        month_filter = st.selectbox("Month", ["All"] + months)

    with col4:
        scripture_filter = st.selectbox("Scripture", ["All"] + scripture_options)

    with col5:
        results_per_page = st.selectbox("Results", [5,10,15])

    if query or scripture_filter != "All":

        query_lower = query.lower()
        query_tokens = tokenize(query_lower)

        sermon_scores = defaultdict(float)
        sermon_chunks = defaultdict(list)

        query_embedding = model.encode([query if query else scripture_filter]).astype("float32")
        D, I = index.search(query_embedding, 200)

        for rank, idx in enumerate(I[0]):

            result = metadata[idx]

            if year_filter != "All" and result["year"] != year_filter:
                continue

            if month_filter != "All" and result["month"] != month_filter:
                continue

            if scripture_filter != "All":
                if scripture_filter not in result.get("scriptures", []):
                    continue

            text_lower = result["text"].lower()
            title_lower = result["title"].lower()

            score = 0

            if query:
                if query_lower in title_lower:
                    score += 5
                if query_lower in text_lower:
                    score += 4

                if len(query_tokens) > 1:
                    if all(token in text_lower for token in query_tokens):
                        score += 5

                token_matches = sum(token in text_lower for token in query_tokens)
                score += token_matches * 2

            for scripture in result.get("scriptures", []):
                if query_lower in scripture.lower():
                    score += 6

            semantic_score = 1 / (1 + D[0][rank])
            if semantic_score > 0.60:
                score += semantic_score * 2

            if score < 2.5:
                continue

            sermon_id = result["title"] + result["date"]
            sermon_scores[sermon_id] += score
            sermon_chunks[sermon_id].append((score, result))

        if not sermon_scores:
            st.error("No relevant results found.")
        else:
            ranked = sorted(sermon_scores.items(), reverse=True, key=lambda x: x[1])
            st.success(f"{len(ranked)} relevant sermons found")

            total_pages = max(1, (len(ranked) + results_per_page - 1) // results_per_page)
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

            start_index = (page_num - 1) * results_per_page
            end_index = start_index + results_per_page

            for sermon_id, _ in ranked[start_index:end_index]:

                chunks = sorted(sermon_chunks[sermon_id], reverse=True, key=lambda x: x[0])
                top_chunk = chunks[0][1]

                youtube_url = f"https://www.youtube.com/watch?v={top_chunk['youtube_id']}&start={int(top_chunk['start'])}"
                formatted_time = format_timestamp(top_chunk["start"])
                snippet = highlight(top_chunk["text"][:400] + "...", query_tokens)

                themes = detect_themes(top_chunk["text"])

                related = []
                for m in metadata:
                    if m["title"] != top_chunk["title"]:
                        if set(m.get("scriptures", [])) & set(top_chunk.get("scriptures", [])):
                            related.append(m["title"])
                related = list(set(related))[:3]

                st.markdown(f"""
                <div class="result-card">
                <h3>{top_chunk['title']}</h3>
                <p><strong>Date:</strong> {top_chunk['date']} |
                <strong>Event:</strong> {top_chunk['event']}</p>
                {f"<p><strong>Themes:</strong> {', '.join(themes)}</p>" if themes else ""}
                <p>{snippet}</p>
                {f"<p><strong>Related Sermons:</strong> {', '.join(related)}</p>" if related else ""}
                <a href="{youtube_url}" target="_blank" class="watch-button">
                ▶ Watch on YouTube (Start at {formatted_time})
                </a>
                </div>
                """, unsafe_allow_html=True)

# =====================================================
# SERMON LIBRARY
# =====================================================

if page == "Sermon Library":

    st.markdown("## Complete Sermon Archive")

    unique_sermons = {}
    for entry in metadata:
        key = entry["title"] + entry["date"]
        if key not in unique_sermons:
            unique_sermons[key] = entry

    grouped = defaultdict(list)
    for sermon in unique_sermons.values():
        grouped[detect_series(sermon["title"])].append(sermon)

    for series_name, sermons in grouped.items():

        st.markdown(f"### {series_name}")

        sorted_sermons = sorted(
            sermons,
            key=lambda x: datetime.strptime(x["date"], "%m/%d/%Y"),
            reverse=True
        )

        for sermon in sorted_sermons:
            youtube_url = f"https://www.youtube.com/watch?v={sermon['youtube_id']}"
            st.markdown(f"""
            <div class="result-card">
            <h4>{sermon['title']}</h4>
            <p><strong>Date:</strong> {sermon['date']} |
            <strong>Event:</strong> {sermon['event']}</p>
            <a href="{youtube_url}" target="_blank" class="watch-button">
            ▶ Watch Full Sermon
            </a>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# SCRIPTURE EXPLORER
# =====================================================

if page == "Scripture Explorer":

    st.markdown("## Scripture Explorer")

    scripture_map = defaultdict(list)
    for m in metadata:
        for s in m.get("scriptures", []):
            scripture_map[s].append(m)

    selected_scripture = st.selectbox(
        "Select Scripture",
        sorted(scripture_map.keys())
    )

    if selected_scripture:

        sermons = scripture_map[selected_scripture]

        unique = {}
        for s in sermons:
            key = s["title"] + s["date"]
            if key not in unique:
                unique[key] = s

        for sermon in unique.values():
            youtube_url = f"https://www.youtube.com/watch?v={sermon['youtube_id']}"
            st.markdown(f"""
            <div class="result-card">
            <h4>{sermon['title']}</h4>
            <p><strong>Date:</strong> {sermon['date']} |
            <strong>Event:</strong> {sermon['event']}</p>
            <a href="{youtube_url}" target="_blank" class="watch-button">
            ▶ Watch Full Sermon
            </a>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# HIDDEN ADMIN DASHBOARD
# =====================================================

if "admin" in st.query_params and st.query_params["admin"] == "true":

    st.markdown("---")
    password = st.text_input("Admin Password", type="password")

    if password == "ufic_admin":

        st.markdown("## Admin Dashboard")

        total_sermons = len(set([m["title"] + m["date"] for m in metadata]))
        total_segments = len(metadata)

        all_scriptures = []
        theme_counter = Counter()

        for m in metadata:
            all_scriptures.extend(m.get("scriptures", []))
            for t in detect_themes(m["text"]):
                theme_counter[t] += 1

        st.metric("Total Sermons", total_sermons)
        st.metric("Total Transcript Segments", total_segments)
        st.metric("Total Scripture References", len(all_scriptures))

        if all_scriptures:
            most_common_scripture = Counter(all_scriptures).most_common(1)[0]
            st.write(f"Most Referenced Scripture: {most_common_scripture[0]} ({most_common_scripture[1]})")

        if theme_counter:
            st.markdown("### Top Themes")
            for theme, count in theme_counter.most_common(5):
                st.write(f"{theme}: {count}")

    elif password:
        st.error("Incorrect password.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer-box">
This tool assists with sermon navigation and structured study.<br>
It does not replace spiritual discernment or personal revelation.
</div>
""", unsafe_allow_html=True)