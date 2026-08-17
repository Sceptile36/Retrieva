import hashlib
import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
from google import genai



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_BASE_DIR = os.path.join(
    BASE_DIR,
    "knowledge_base"
)

os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)


# -----------------------------------
# Page setup
# -----------------------------------

st.set_page_config(
    page_title="Telecom RAG Assistant",
    page_icon="📡"
)

st.title("📡 Telecom RAG Assistant")
st.write("Upload telecom documents and ask questions about them.")


# -----------------------------------
# Load Gemini API
# -----------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# -----------------------------------
# Load embedding model
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# Connect to ChromaDB
# -----------------------------------

vector_store = Chroma(
    collection_name="telecom_documents",
    embedding_function=embeddings,
    persist_directory="D:/Vs Code Python/python1/AI_chatbot_RAG_LANGCHAIN/data/chroma_db"
)


# -----------------------------------
# Upload PDFs
# -----------------------------------

uploaded_files = st.file_uploader(
    "Upload telecom PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)


# -----------------------------------
# Process uploaded PDFs
# -----------------------------------

if uploaded_files:

    if st.button("Process Documents"):

        all_documents = []

        for uploaded_file in uploaded_files:

            # Save PDF permanently inside knowledge_base
            save_path = os.path.join(
                KNOWLEDGE_BASE_DIR,
                uploaded_file.name
            )

            # Save the uploaded PDF
            with open(save_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            # Load PDF
            loader = PyPDFLoader(save_path)

            documents = loader.load()

            # Add filename to metadata
            for document in documents:
                document.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)


        # -----------------------------------
        # Split into chunks
        # -----------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(
            all_documents
        )


        # -----------------------------------
        # Create unique IDs
        # -----------------------------------

        chunk_ids = []

        for i, chunk in enumerate(chunks):

            source = chunk.metadata.get(
                "source",
                "unknown"
            )

            page = chunk.metadata.get(
                "page",
                0
            )

            chunk_id = hashlib.md5(
                f"{source}_{page}_{i}".encode()
            ).hexdigest()

            chunk_ids.append(chunk_id)


        # -----------------------------------
        # Check which chunks already exist
        # -----------------------------------

        existing = vector_store.get(
            ids=chunk_ids
        )

        existing_ids = set(
            existing["ids"]
        )


        # -----------------------------------
        # Keep only new chunks
        # -----------------------------------

        new_chunks = []
        new_ids = []

        for chunk, chunk_id in zip(
            chunks,
            chunk_ids
        ):

            if chunk_id not in existing_ids:

                new_chunks.append(chunk)
                new_ids.append(chunk_id)


        # -----------------------------------
        # Add only new chunks
        # -----------------------------------

        if new_chunks:

            vector_store.add_documents(
                documents=new_chunks,
                ids=new_ids
            )

            st.success(
                f"Added {len(new_chunks)} new chunks "
                f"from {len(uploaded_files)} document(s)!"
            )

        else:

            st.info(
                "All uploaded documents are already "
                "stored in ChromaDB."
            )

# -----------------------------------
# Ask question
# -----------------------------------

question = st.text_input(
    "Ask a question:"
)


# -----------------------------------
# Ask Gemini
# -----------------------------------

if st.button("Ask"):

    if not question:

        st.warning("Please enter a question.")

    else:

        # Retrieve relevant chunks
        results = vector_store.similarity_search(
            question,
            k=10
        )


        # -----------------------------------
        # Display retrieved sources
        # -----------------------------------

        st.subheader("📚 Retrieved Sources")

        shown_sources = set()

        for result in results:

            source = result.metadata.get(
                "source",
                "Unknown"
            )

            if source not in shown_sources:

                st.write(f"📄 {source}")

                shown_sources.add(source)


        # -----------------------------------
        # Create context
        # -----------------------------------

        context = "\n\n".join(

            f"Source: {result.metadata.get('source')}\n"
            f"Page: {result.metadata.get('page')}\n"
            f"Content: {result.page_content}"

            for result in results
        )


        # -----------------------------------
        # Gemini prompt
        # -----------------------------------

        prompt = f"""
Answer the question using only the context provided below.

If the context does not contain enough information
to answer the question, say that the information
was not found in the uploaded documents.

Context:
{context}

Question:
{question}
"""


        # -----------------------------------
        # Ask Gemini
        # -----------------------------------

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )


        # -----------------------------------
        # Display answer
        # -----------------------------------

        st.subheader("🤖 Gemini Answer")

        st.write(response.text)


        # -----------------------------------
        # Display chunks
        # -----------------------------------

        st.subheader("🔎 Retrieved Chunks")

        for i, result in enumerate(results):

            with st.expander(
                f"Chunk {i + 1}"
            ):

                st.write(
                    result.page_content
                )

                st.write(
                    "Source:",
                    result.metadata.get("source")
                )

                st.write(
                    "Page:",
                    result.metadata.get("page")
                )
