# LangChain RAG Repository

## 📌 Overview
This repository contains various implementations of **Retrieval-Augmented Generation (RAG)** using **LangChain, LangGraph, OpenAI, Pinecone, and Wikipedia**. It also includes a **human-in-the-loop** implementation for enhanced AI-assisted workflows.

--

## 📂 Files in the Repository

### 1️⃣ **Lchain.py**
   - 📖 Covers the **basics of LangChain**.
   - 🔗 Demonstrates how to create and manage chains for AI-driven workflows.

### 2️⃣ **Lgraph.py**
   - 🤖 Implements **RAG (Retrieval-Augmented Generation) using LangGraph**.
   - 📊 Utilizes LangGraph to build a structured retrieval and response pipeline.

### 3️⃣ **QnA.py**
   - 🧠 Implements **RAG using OpenAI and Pinecone**.
   - 🔍 Uses OpenAI’s LLM for question-answering and Pinecone for vector storage.
   - 📌 Performs semantic search to retrieve relevant information efficiently.

### 4️⃣ **LHuman.py**
   - 👥 Implements a **human-in-the-loop** RAG system using LangGraph.
   - ✅ Allows human intervention to validate and refine AI-generated responses.
   
### 5️⃣ **wiki.py**
   - 🌍 Implements **RAG using LangChain and Wikipedia**.
   - 📚 Retrieves information directly from Wikipedia for knowledge augmentation.

---

## 🛠 Setup Instructions

### 1️⃣ **Clone the repository:**
```sh
git clone https://github.com/RAG_AI.git

```

### 2️⃣ **Install dependencies:**
```sh
pip install -r requirements.txt
```

### 3️⃣ **Set up API keys:**
Create a `.env` file and add the necessary keys for OpenAI, Pinecone, etc.
```sh
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 4️⃣ **Run the scripts:**
```sh
python Lgraph.py
```

---

## 🚀 Usage
- 🔎 Use `QnA.py` for a **Q&A system** powered by OpenAI and Pinecone.
- 🏗 Run `LHuman.py` for **human-in-the-loop AI interaction**.
- 📖 Try `wiki.py` for **Wikipedia-based RAG**.

---

## 🤝 Contributing
Feel free to contribute by creating pull requests or submitting issues for improvements!

---

## 📜 License
This project is licensed under the **MIT License**.


