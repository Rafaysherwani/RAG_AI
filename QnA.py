import os
import asyncio
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Set your API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"

# Initialize OpenAI


# Initialize Pinecone
pc = Pinecone()

# Check if the index exists; if not, create it
INDEX_NAME = "python-index"
EMBEDDING_DIMENSION = 384  # Adjust based on your embedding model

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load the embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Function to embed and upsert data into Pinecone
async def embed_and_upsert():
    # Example data
    data = [
        {"id": "1", "text": "Python is a programming language."},
        {"id": "2", "text": "It is widely used for web development, data science, and AI."}
    ]

    # Generate embeddings
    embeddings = embedder.encode([item["text"] for item in data])

    # Prepare data for upsert, including metadata
    upsert_data = [
        {
            "id": item["id"],
            "values": embedding,
            "metadata": {"text": item["text"]}
        }
        for item, embedding in zip(data, embeddings)
    ]

    # Debug: Print upsert data
    for vector in upsert_data:
        print("Upserting Vector:", vector)

    # Upsert data into Pinecone
    index.upsert(upsert_data)
    print("Data inserted into Pinecone successfully!")

    # Verify data insertion
    print("Index Stats:", index.describe_index_stats())
async def embed_and_upsert():
    # Example data
    data = [
        {"id": "1", "text": "Python is a programming language."},
        {"id": "2", "text": "It is widely used for web development, data science, and AI."}
    ]

    # Generate embeddings
    embeddings = embedder.encode([item["text"] for item in data])

    # Prepare data for upsert, including metadata
    upsert_data = [
        {
            "id": item["id"],
            "values": embedding,
            "metadata": {"text": item["text"]}
        }
        for item, embedding in zip(data, embeddings)
    ]

    # Debug: Print upsert data
    for vector in upsert_data:
        print("Upserting Vector:", vector)

    # Upsert data into Pinecone
    index.upsert(upsert_data)
    print("Data inserted into Pinecone successfully!")

    # Verify data insertion
    print("Index Stats:", index.describe_index_stats())

# Function to perform RAG-based QA
async def rag_qa(query):
    # Generate embedding for the query
    query_embedding = embedder.encode(query)

    # Search in Pinecone
    search_results = index.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True)

    # Debug: Check the structure of search_results
    print("Search Results:", search_results)
    
    # Safely extract metadata and handle partial matches
    contexts = []
    for result in search_results.get("matches", []):
        metadata = result.get("metadata", {})
        text = metadata.get("text", None)  # Adjust to match the metadata key
        if text:
            contexts.append(text)
        else:
            print(f"Match with ID {result['id']} has no 'text' metadata.")

    if not contexts:
        return "No relevant context found in the Pinecone index."

    # Prepare prompt for OpenAI API
    prompt = (
        "Answer the following question based on the context below. If the answer is not in the context, say so.\n\n"
        "Context:\n"
        f"{' '.join(contexts)}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # Use OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Main function
async def main():
    # Step 1: Upsert data into Pinecone
    await embed_and_upsert()

    # Step 2: Query the system
    query = "What is Python?"
    answer = await rag_qa(query)
    print(f"Q: {query}\nA: {answer}")


if __name__ == "__main__":
    asyncio.run(main())