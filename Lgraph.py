from dotenv import load_dotenv
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langgraph.agent import GraphAgent
from langgraph.utils import build_knowledge_graph
from langchain.evaluation.qa import QAEvaluator
from sklearn.metrics import accuracy_score

load_dotenv()

# Retrieve keys
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Task 1: Build Agentic RAG with Evaluation Procedure
# ---------------------------------------------
# Step 1: Initialize the Vector Store and Embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("path_to_your_vectorstore", embeddings)

# Step 2: Initialize the LLM
llm = OpenAI(model="gpt-4")

# Step 3: Define RAG-based Retrieval Pipeline
retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# Step 4: Create Knowledge Graph for Agent
knowledge_graph = build_knowledge_graph("knowledge_data.json")  # Provide your JSON data.
agent = GraphAgent(knowledge_graph, retrieval_qa_chain)

# Step 5: RAG Agent Functionality
def rag_agent_query(query):
    response = agent.run(query)
    return response


# Task 2: Build a Basic RAG Pipeline
# ----------------------------------
# Step 1: Basic Retrieval Pipeline
def build_basic_rag_pipeline(query, vector_store, llm):
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)
    
    # Generate answers based on retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    answer = llm("Answer the question based on this context: \n" + context + f"\n\nQuestion: {query}")
    return answer

# Example Usage
query = "What is the capital of France?"
answer = build_basic_rag_pipeline(query, vector_store, llm)
print("Basic RAG Answer:", answer)


# Task 3: Evaluation Procedure for RAG Results
# ---------------------------------------------
def evaluate_rag_results(test_data, vector_store, llm):
    evaluator = QAEvaluator()

    y_true = []
    y_pred = []

    for query, expected_answer in test_data.items():
        # Use the retrieval-augmented generation pipeline
        prediction = build_basic_rag_pipeline(query, vector_store, llm)

        # Evaluate and store the results
        y_true.append(expected_answer)
        y_pred.append(prediction)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    print("RAG Evaluation Accuracy:", accuracy)

    # Optionally, return detailed evaluation results
    return {
        "accuracy": accuracy,
        "true_answers": y_true,
        "predicted_answers": y_pred
    }

# Example Evaluation Dataset
test_data = {
    "What is the capital of France?": "Paris",
    "Who wrote '1984'?": "George Orwell",
}

# Evaluate
results = evaluate_rag_results(test_data, vector_store, llm)
print("Evaluation Results:", results)

