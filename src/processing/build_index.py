import os
import re
import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
from collections import Counter

# =====================================================
# CONFIGURATION
# =====================================================

CHUNK_DIR = "data/processed/transcripts/chunks"
INDEX_DIR = "data/index"
METADATA_FILE = "data/metadata/sermon_metadata.csv"

os.makedirs(INDEX_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# SCRIPTURE DETECTION
# =====================================================

BIBLE_BOOKS = [
    "genesis","exodus","leviticus","numbers","deuteronomy",
    "joshua","judges","ruth","1 samuel","2 samuel",
    "1 kings","2 kings","1 chronicles","2 chronicles",
    "ezra","nehemiah","esther","job","psalms","proverbs",
    "ecclesiastes","song of solomon","isaiah","jeremiah",
    "lamentations","ezekiel","daniel","hosea","joel","amos",
    "obadiah","jonah","micah","nahum","habakkuk","zephaniah",
    "haggai","zechariah","malachi","matthew","mark","luke",
    "john","acts","romans","1 corinthians","2 corinthians",
    "galatians","ephesians","philippians","colossians",
    "1 thessalonians","2 thessalonians","1 timothy","2 timothy",
    "titus","philemon","hebrews","james","1 peter","2 peter",
    "1 john","2 john","3 john","jude","revelation"
]

def extract_scriptures(text):
    text_lower = text.lower()
    found = []

    # Detect book names
    for book in BIBLE_BOOKS:
        if book in text_lower:
            found.append(book.title())

    # Detect chapter:verse (e.g., 9:27)
    verse_pattern = r"\b\d{1,3}:\d{1,3}\b"
    verses = re.findall(verse_pattern, text)

    return list(set(found + verses))

# =====================================================
# MUSIC / NON-SERMON FILTER
# =====================================================

def is_music_segment(text):
    music_keywords = [
        "music", "sing", "worship", "choir",
        "hallelujah hallelujah", "repeat after me"
    ]

    text_lower = text.lower()
    music_score = sum(keyword in text_lower for keyword in music_keywords)

    words = text_lower.split()
    if len(words) == 0:
        return True

    unique_ratio = len(set(words)) / len(words)

    # repetitive low-meaning text (singing)
    if music_score >= 2 or unique_ratio < 0.35:
        return True

    return False

# =====================================================
# LOAD METADATA
# =====================================================

metadata_df = pd.read_csv(METADATA_FILE)
metadata_df.columns = metadata_df.columns.str.strip()

# =====================================================
# BUILD INDEX
# =====================================================

all_chunks = []
all_texts = []

for filename in os.listdir(CHUNK_DIR):

    if not filename.endswith(".json"):
        continue

    sermon_id = filename.split("_")[0] + "_" + filename.split("_")[1]

    sermon_info = metadata_df[metadata_df["SERMON ID"] == sermon_id]

    if sermon_info.empty:
        continue

    sermon_info = sermon_info.iloc[0]

    raw_date = sermon_info["DATE"]
    parsed_date = datetime.strptime(raw_date, "%m/%d/%Y")

    month_name = parsed_date.strftime("%B").lower()
    year = parsed_date.strftime("%Y")

    path = os.path.join(CHUNK_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for chunk in data["chunks"]:

        text = chunk["text"]

        # Skip music / noise segments
        if is_music_segment(text):
            continue

        scriptures = extract_scriptures(text)

        combined_text = f"""
        {sermon_info['TITLE']}
        {sermon_info['EVENT']}
        {month_name}
        {year}
        {text}
        """

        all_texts.append(combined_text)

        all_chunks.append({
            "filename": filename,
            "title": sermon_info["TITLE"],
            "date": raw_date,
            "event": sermon_info["EVENT"],
            "month": month_name,
            "year": year,
            "youtube_id": str(sermon_info["YOUTUBE_ID"]).strip(),
            "start": chunk["start"],
            "end": chunk["end"],
            "text": text,
            "scriptures": scriptures
        })

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

print("Generating embeddings...")

embeddings = model.encode(all_texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# =====================================================
# CREATE FAISS INDEX
# =====================================================

print("Creating FAISS index...")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(INDEX_DIR, "sermon_index.faiss"))

# =====================================================
# SAVE METADATA
# =====================================================

with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

# =====================================================
# ANALYTICS OUTPUT
# =====================================================

all_scriptures = []
for chunk in all_chunks:
    all_scriptures.extend(chunk["scriptures"])

if all_scriptures:
    most_common = Counter(all_scriptures).most_common(5)
    print("\nTop Scripture Mentions:")
    for scripture, count in most_common:
        print(f"{scripture}: {count} times")

print("\nIndex rebuilt successfully.")