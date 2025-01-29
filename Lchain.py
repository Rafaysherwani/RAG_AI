from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_core.runnables import RunnableSequence
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

import os
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define the prompt template
prompt = PromptTemplate.from_template("What is the capital of {country}?")

# Initialize the LLM
llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.3)

# Create a sequence (chain) with the prompt and LLM
chain = RunnableSequence(first=prompt, last=llm)

# Execute the chain
result = chain.invoke({"country": "France"})

# Print the result
print(result)
