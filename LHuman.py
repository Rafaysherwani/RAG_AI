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
from langgraph.graph import StateGraph
from typing import Dict, Any
load_dotenv()

# Get API keys from environment variables

os.environ["PINECONE_API_KEY"] = os.get.env("PINECONE_API_KEY")
os.environ["PINECONE_ENV"] = os.get.env("PINECONE_ENV")
os.environ["OPENAI_API_KEY"] = os.get.env("OPENAI_API_KEY")

def initialize_pinecone(index_name):
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    dimension = 1536
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.environ.get("PINECONE_ENV")
            )
        )

def load_wikipedia_content(topic: str):
    loader = WikipediaLoader(query=topic, lang="en")
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_pinecone_vector_store(index_name, documents):
    embeddings = OpenAIEmbeddings()
    vector_store = LangChainPinecone.from_documents(documents, embeddings, index_name=index_name)
    return vector_store

def generate_ai_answer(state: Dict[str, Any]):
    retriever = state["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    docs = retriever.get_relevant_documents(state["question"])
    context = " ".join([doc.page_content for doc in docs])
    ai_answer = llm(prompt.format(context=context, question=state["question"]))
    state["ai_answer"] = ai_answer
    return state

def human_review(state: Dict[str, Any]):
    print("AI-generated Answer: ", state["ai_answer"])
    human_feedback = input("Do you want to modify the answer? (yes/no): ")
    if human_feedback.lower() == "yes":
        human_answer = input("Enter your modified answer: ")
        state["final_answer"] = human_answer
    else:
        state["final_answer"] = state["ai_answer"]
    return state

def run_rag_with_wikipedia(topic, question, index_name):
    initialize_pinecone(index_name)
    documents = load_wikipedia_content(topic)
    split_docs = split_documents(documents)
    vector_store = create_pinecone_vector_store(index_name, split_docs)
    
    graph = StateGraph(Dict[str, Any])  # Define state as a dictionary
    graph.add_node("generate_ai_answer", generate_ai_answer)
    graph.add_node("human_review", human_review)
    graph.set_entry_point("generate_ai_answer")
    graph.add_edge("generate_ai_answer", "human_review")
    
    executor = graph.compile()
    initial_state = {"vector_store": vector_store, "question": question}
    final_state = executor.invoke(initial_state)
    return final_state["final_answer"]

if __name__ == "__main__":
    index_name = "example-index"
    topic = "Artificial Intelligence"
    question = "What is AI?"
    answer = run_rag_with_wikipedia(topic, question, index_name)
    print("Final Answer:", answer)
