# Retrieva

AI-powered document question-answering system using Retrieval-Augmented Generation (RAG).

Retrieva allows users to upload PDF documents and ask natural-language questions about their content. The system retrieves the most relevant document chunks using semantic similarity and provides context-aware answers through an LLM.

## Features

- Upload multiple PDF documents
- Automatic document loading and text extraction
- Recursive text chunking
- Hugging Face sentence embeddings
- Semantic similarity search using ChromaDB
- Retrieval-Augmented Generation (RAG)
- Source and page tracking for retrieved information
- LLM-powered answers using Google Gemini
- Streamlit-based web interface
- Duplicate document detection

## Architecture

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Chunking
      ↓
Hugging Face Embeddings
      ↓
ChromaDB Vector Store
      ↓
Similarity Search
      ↓
Relevant Chunks
      ↓
Gemini LLM
      ↓
Generated Answer
````

## Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Google Gemini API
* Streamlit
* PyPDF

## How It Works

1. User uploads one or more PDF documents.
2. The PDFs are loaded and divided into smaller text chunks.
3. Each chunk is converted into a numerical vector using a Hugging Face embedding model.
4. The vectors are stored in ChromaDB.
5. When a user asks a question, the question is converted into an embedding.
6. ChromaDB retrieves the most semantically similar chunks.
7. The retrieved context is passed to Gemini.
8. Gemini generates an answer based on the retrieved document content.

## Installation

Clone the repository:

```bash
git clone https://github.com/Sceptile36/Retrieva.git
cd Retrieva
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run rag_pipeline(front+backend).py
```

## Project Structure

```text
Retrieva/
│
├── rag_pipeline(front+backend)
├── retrieval_engine(backend_understanding)
├── requirements.txt
├── .gitignore
│
├── data/
│   └── chroma_db/
│
├── knowledge_base/
│
└── src/
```

## Future Improvements

* Improved retrieval using hybrid search
* Reranking retrieved chunks
* Conversation memory
* Support for additional document formats
* Retrieval evaluation and relevance metrics
* Improved citation and source presentation
