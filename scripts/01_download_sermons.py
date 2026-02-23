
from pathlib import Path
import subprocess

print("=== UFIC INGESTION SCRIPT STARTED ===")

# Base directory for audio downloads
AUDIO_DIR = Path("data/raw/youtube/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def download_youtube_audio(url):
    """
    Downloads audio from a YouTube video or playlist as MP3.
    """
    command = [
    "yt-dlp",
    "--cookies", "youtube_cookies.txt",
    "-f", "bestaudio",
    "--extract-audio",
    "--audio-format", "mp3",
    "-o", str(AUDIO_DIR / "%(upload_date)s_%(title)s.%(ext)s"),
    url
]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    print("Waiting for YouTube URL input...")
    youtube_url = input("Paste YouTube video or playlist URL: ").strip()
    download_youtube_audio(youtube_url)
