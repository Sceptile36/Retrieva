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
