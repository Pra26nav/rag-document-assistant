# RAG Document Assistant

Intelligent PDF Q&A system powered by LangChain, FAISS, and Groq.

## Architecture
PDF Upload (single or multi)
|
PyPDF -> Text Extraction
|
RecursiveCharacterTextSplitter
|
HuggingFace Embeddings (all-MiniLM-L6-v2, local)
|
FAISS Vector Index + MD5 Cache
|
User Question
|
FAISS similarity_search (k=4)
|
Context + History -> Groq LLM
|
Answer + Source Citations + Confidence Score

## Tech Stack
- LangChain + FAISS — vector search
- HuggingFace — local embeddings (no API cost)
- Groq (Llama 3.3 70B) — LLM inference
- Streamlit — UI
- PyPDF — PDF extraction

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Add `.env`:
GROQ_API_KEY=your_key_here
Run:
```bash
streamlit run app.py
```

## Features
- Multi-PDF support with per-document tab switching
- Smart MD5 caching — same PDF loads instantly
- Source citations with page numbers
- Answer confidence score
- Conversational memory (last 3 exchanges)
- Export chat as .txt
- Batch embedding for faster indexing
- Per-page progress tracking

