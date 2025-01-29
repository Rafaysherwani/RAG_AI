import os
import pinecone
from typing import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Get API keys from environment variables

os.environ["PINECONE_API_KEY"] = os.get.env("PINECONE_API_KEY")
os.environ["PINECONE_ENV"] = os.get.env("PINECONE_ENV")
os.environ["OPENAI_API_KEY"] = os.get.env("OPENAI_API_KEY")

# Initialize Pinecone
def initialize_pinecone(index_name):
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    return pc

class State(TypedDict):
    query: str
    embedding: list
    context: str
    response: str

# Initialize components
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()

# Initialize Pinecone
index_name = "example-index"
dimension = 1536  # OpenAI embedding size

pc = initialize_pinecone(index_name)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.environ.get("PINECONE_ENV")
        )
    )

index = pc.Index(index_name)

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Initialize the StateGraph
graph_builder = StateGraph(State)

# 1. Embedding Node
def embedding_node(state: State):
    """Generate embedding for the query."""
    state["embedding"] = embeddings.embed_query(state["query"])
    return state

graph_builder.add_node("embed_query", embedding_node)

# 2. Retrieval Node
def retrieval_node(state: State):
    """Retrieve relevant documents from Pinecone."""
    results = vectorstore.similarity_search(state["query"], k=3)
    state["context"] = " ".join([doc.page_content for doc in results])
    return state

graph_builder.add_node("retrieve_context", retrieval_node)

# 3. Processing Node
def processing_node(state: State):
    """Process or clean the retrieved context."""
    max_length = 1000
    state["context"] = state["context"][:max_length]
    return state

graph_builder.add_node("process_context", processing_node)

# 4. Generation Node
def generation_node(state: State):
    """Generate a response using the LLM."""
    prompt = (
        "Answer the following question based on the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context: {state['context']}\n\nQuestion: {state['query']}\nAnswer:"
    )
    response = llm.invoke(prompt)
    
    # Extract only the actual text response
    state["response"] = response.content if hasattr(response, "content") else str(response)
    
    return state

graph_builder.add_node("generate_response", generation_node)

# Define Edges
graph_builder.add_edge(START, "embed_query")
graph_builder.add_edge("embed_query", "retrieve_context")
graph_builder.add_edge("retrieve_context", "process_context")
graph_builder.add_edge("process_context", "generate_response")
graph_builder.add_edge("generate_response", END)

# Compile the graph
graph = graph_builder.compile()

def query_rag(query: str) -> dict:
    """Function to query the RAG system"""
    initial_state = {
        "query": query,
        "embedding": [],
        "context": "",
        "response": ""
    }
    return graph.invoke(initial_state)

if __name__ == "__main__":
    result = query_rag("What is AI?")
    print(f"Query: What is AI?")
    print(f"Response: {result['response']}")