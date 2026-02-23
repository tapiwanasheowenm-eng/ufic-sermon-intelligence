\# UFIC Sermon Search System – Architecture Overview



\## Purpose

This system provides a searchable index of sermons preached by Prophet Emmanuel Makandiwa,

allowing users to quickly locate specific teachings by keyword, phrase, topic, date, or event.



The system is designed as a navigation and retrieval tool, not a source of interpretation,

doctrinal explanation, or spiritual guidance.



\## Core Principles

\- The system does not generate teachings or answers.

\- The system does not summarize or reinterpret sermons.

\- The system only points users to original sermon content.

\- Spiritual understanding and discernment remain the responsibility of the listener.



\## High-Level Architecture



1\. Content Sources

   - Public sermons (e.g., YouTube)

   - Church-hosted sermon archives

   - Other authorized sermon platforms



2\. Processing Pipeline

   - Audio extraction

   - Speech-to-text transcription with timestamps

   - Text cleaning and normalization

   - Chunking into time-aligned segments

   - Semantic embedding for search



3\. Storage

   - Raw audio files (unaltered)

   - Processed transcripts (JSON + text)

   - Metadata database (SQLite)

   - Vector index for semantic search



4\. Search Layer

   - Keyword-based search

   - Semantic (meaning-based) search

   - Hybrid ranking of results



5\. Application Layer

   - Web-based interface

   - Low-bandwidth friendly

   - Links users directly to original sermon timestamps



\## Ethical Boundary

This system exists to assist discovery, study, and review of teachings.

It is not a replacement for the Holy Spirit, personal study, prayer, or pastoral guidance.



\## Future Extensions (Optional)

\- Automated ingestion of new sermons

\- Multi-language support

\- Mobile-first interface

