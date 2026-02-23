import os
import json
import re

RAW_DIR = "data/processed/transcripts/json"
CLEAN_DIR = "data/processed/transcripts/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)


def normalize_text(text):
    # Convert to lowercase
    text = text.lower()

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_file(filename):
    raw_path = os.path.join(RAW_DIR, filename)
    clean_path = os.path.join(CLEAN_DIR, filename)

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_segments = []

    for segment in data.get("segments", []):
        text = segment.get("text", "")
        normalized = normalize_text(text)

        if normalized:
            cleaned_segment = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": normalized
            }
            cleaned_segments.append(cleaned_segment)

    cleaned_data = {
        "text": normalize_text(data.get("text", "")),
        "segments": cleaned_segments
    }

    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"Cleaned: {filename}")


def main():
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".json"):
            clean_file(filename)


if __name__ == "__main__":
    main()
