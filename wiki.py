from re import L
import os
import wikipedia
import pinecone
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

# Get API keys from environment variables

os.environ["PINECONE_API_KEY"] = os.get.env("PINECONE_API_KEY")
os.environ["PINECONE_ENV"] = os.get.env("PINECONE_ENV")
os.environ["OPENAI_API_KEY"] = os.get.env("OPENAI_API_KEY")

# Step 1: Initialize Pinecone
def initialize_pinecone(index_name):

    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )

    dimension = 1536  # Dimension for OpenAI embeddings

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.environ.get("PINECONE_ENV")  # Use the environment variable
            )
        )

# Step 2: Load Wikipedia content
def load_wikipedia_content(topic: str):
    loader = WikipediaLoader(query=topic, lang="en")
    documents = loader.load()
    return documents

# Step 3: Split documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]  # Hierarchical splitting
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Step 4: Create a Pinecone vector store
def create_pinecone_vector_store(index_name, documents):
    embeddings = OpenAIEmbeddings()
    vector_store =LangChainPinecone.from_documents(documents, embeddings, index_name=index_name)
    return vector_store

# Step 5: Set up a custom QA pipeline
def create_custom_qa_pipeline(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = OpenAI(temperature=0)  # Adjust temperature for creativity
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def qa_pipeline(question):
        docs = retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in docs])
        answer = llm(prompt.format(context=context, question=question))
        return answer

    return qa_pipeline

# Main function to run RAG
def run_rag_with_wikipedia(topic, question, index_name):
    # Initialize Pinecone
    initialize_pinecone(index_name)

    # Load and process Wikipedia content
    documents = load_wikipedia_content(topic)
    split_docs = split_documents(documents)

    # Create vector store
    vector_store = create_pinecone_vector_store(index_name, split_docs)

    # Set up QA pipeline
    qa_pipeline = create_custom_qa_pipeline(vector_store)

    # Ask a question
    answer = qa_pipeline(question)
    return answer

# Example usage
if __name__ == "__main__":
    # Set Pinecone index name
    index_name = "example1-index"

    # Example topic and question
    topic = "Quantum Computing"
    question = "What are the applications of quantum computing?"

    # Run RAG pipeline
    answer = run_rag_with_wikipedia(topic, question, index_name)
    print("Answer:", answer)
