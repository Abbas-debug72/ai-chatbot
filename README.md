# Knowledge-Based RAG Chatbot

An intelligent chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from PDF documents.

## Features

- 📄 Ingests multiple PDFs into a searchable knowledge base
- 🔍 Intelligent retrieval with document type detection
- 🧠 FAISS vector search + Groq LLM for accurate answers
- 💬 Persistent conversation memory
- 🎯 Document focus mode (query specific files)
- 📊 Source citations with page numbers

## Tech Stack

- Python, Flask, LangChain
- FAISS (Facebook AI Similarity Search)
- Groq LLM (Llama 3.1)
- Sentence Transformers (all-MiniLM-L6-v2)

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add PDFs to the `pdfs/` folder
4. Build the brain: `python ingest_all.py`
5. Set your API key: `export GROQ_API_KEY=gsk_your_key`
6. Run: `python app.py`
7. Open: http://127.0.0.1:5000

## Project Structure

- `brain.py` - Knowledge engine (ingestion, embeddings, search)
- `app.py` - Flask API server
- `memory.py` - Conversation memory
- `ingest_all.py` - One-time PDF ingestion
- `templates/index.html` - Web chat interface
  
