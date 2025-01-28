import os
import pinecone
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone, ServerlessSpec  # Updated import

# Initialize Pinecone client
pc = Pinecone() # Will automatically use PINECONE_API_KEY from environment
# Define the State for the pipeline
class State(TypedDict):
    query: str
    embedding: list
    context: str
    response: str

# Initialize components
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
embeddings = OpenAIEmbeddings()

# Initialize Pinecone with index creation if needed
index_name = "rag"  # Replace with your Pinecone index name
dimension = 1536  # OpenAI embeddings dimension

# Check if index exists and create if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",  # Added comma here
        spec=ServerlessSpec(
            cloud="aws",
            region= "us-east-1"
        )
    )

# Get the index
index = pc.Index(index_name)

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"  # The key which contains the text in your Pinecone index
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
    results = vectorstore.similarity_search_by_vector(state["embedding"], k=3)
    state["context"] = " ".join([doc.page_content for doc in results])
    return state

graph_builder.add_node("retrieve_context", retrieval_node)

# 3. Processing Node
def processing_node(state: State):
    """Process or clean the retrieved context."""
    max_length = 1000  # Adjust as needed
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
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["response"] = response.content
    return state

graph_builder.add_node("generate_response", generation_node)

# Define Edges (Data Flow)
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
    # Example usage
    result = query_rag("What is Python?")
    print(f"Query: What is Python?")
    print(f"Response: {result['response']}")