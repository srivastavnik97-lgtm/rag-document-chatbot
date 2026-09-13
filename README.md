# 📄 RAG Document Chatbot

Ask questions about any PDF and get accurate, grounded answers — powered by Retrieval-Augmented Generation (RAG).

## What this does

Upload a PDF, ask a question in plain English, and get an answer generated from the document's actual content — not the AI's general knowledge. If the answer isn't in the document, it says so instead of guessing.

**Example:** upload an invoice, ask *"What is the total amount?"*, get the exact figure back — pulled from the document itself.

## How it works

1. **Load** — the PDF is read and its text extracted
2. **Chunk** — the text is split into small, overlapping pieces so relevant sections can be found precisely
3. **Embed & store** — each chunk is converted into a vector (a numeric representation of its meaning) and stored in a local vector database (ChromaDB)
4. **Retrieve** — when a question is asked, the most semantically relevant chunks are retrieved
5. **Generate** — those chunks are passed to an LLM (via Groq) along with the question, with an instruction to answer only from the provided context

This is a standard RAG pipeline — the same core pattern used in production document-QA and knowledge-base assistants.

## Tech stack

- **Python**
- **LangChain** — orchestrates the load → chunk → retrieve → generate pipeline
- **ChromaDB** — local vector database for storing and searching document chunks
- **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`) — generates embeddings locally, no API cost
- **Groq API** (`openai/gpt-oss-20b`) — fast LLM inference for the final answer
- **Streamlit** — web interface

## Running it locally

1. Clone this repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root with your own Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
   (Get a free key at [console.groq.com](https://console.groq.com))
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Upload a PDF and start asking questions.

## A real debugging note

Early on, retrieval initially missed the correct chunk for numeric/tabular data (e.g. an invoice's total amount) because the embedding model matched on general semantic similarity rather than exact figures. Increasing the number of retrieved chunks (`k`) resolved it — a good example of how retrieval quality, not just model quality, drives RAG accuracy in practice.

## What I'd improve next

- Support multiple documents at once
- Better chunking strategy for tables/structured data
- Persistent vector storage instead of rebuilding per session
