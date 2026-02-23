import os
import json

CLEAN_DIR = "data/processed/transcripts/cleaned"
CHUNK_DIR = "data/processed/transcripts/chunks"

os.makedirs(CHUNK_DIR, exist_ok=True)

TARGET_DURATION = 45  # seconds per chunk


def chunk_file(filename):
    clean_path = os.path.join(CLEAN_DIR, filename)
    chunk_path = os.path.join(CHUNK_DIR, filename)

    with open(clean_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    chunks = []

    current_chunk = []
    chunk_start = None
    chunk_end = None

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        if chunk_start is None:
            chunk_start = start

        current_chunk.append(text)
        chunk_end = end

        if chunk_end - chunk_start >= TARGET_DURATION:
            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": " ".join(current_chunk)
            })

            current_chunk = []
            chunk_start = None
            chunk_end = None

    # Catch remaining text
    if current_chunk:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": " ".join(current_chunk)
        })

    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

    print(f"Chunked: {filename}")


def main():
    for filename in os.listdir(CLEAN_DIR):
        if filename.endswith(".json"):
            chunk_file(filename)


if __name__ == "__main__":
    main()
