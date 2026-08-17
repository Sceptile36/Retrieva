from unittest import result

from langchain_community.document_loaders import PyPDFLoader

pdf_paths = [
    "knowledge_base/ericsson_whitepaper.pdf",
    "knowledge_base/nokia_whitepaper.pdf",
    "knowledge_base/o_ran_spec.pdf"
]
documents = []

for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    pdf_documents = loader.load()

    documents.extend(pdf_documents)


print("Number of pages loaded:", len(documents))

print("\nFirst page content:")
print(documents[0].page_content[:1000])

print("\nMetadata:")
print(documents[0].metadata)



from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("\nNumber of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst chunk metadata:")
print(chunks[0].metadata)




from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded!")

test_embedding = embeddings.embed_query(
    "What is 5G network slicing?"
)

print("Embedding length:", len(test_embedding))
print("First 10 values:", test_embedding[:10])



from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="telecom_documents",
    embedding_function=embeddings,
    persist_directory="data/chroma_db"
)

#vector_store.add_documents(chunks)
#print("\nDocuments stored in Chroma DB!")


print("Total documents in Chroma:", vector_store._collection.count())


## adding gemini api

from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)




while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    results = vector_store.similarity_search(question, k=3)

    # Show where the chunks came from
    print("\nRETRIEVED SOURCES:")

    for result in results:
        print(result.metadata.get("source"))

    # Combine retrieved chunks into context
    context = "\n\n".join(
        f"Source: {result.metadata.get('source')}\n"
        f"Page: {result.metadata.get('page')}\n"
        f"Content: {result.page_content}"
        for result in results
    )

    # Create prompt for Gemini
    prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}
"""

    # Ask Gemini
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    print("\nGEMINI:")
    print(response.text)
