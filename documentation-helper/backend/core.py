import os
from typing import Any,Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from referencing.typing import Retrieve
from streamlit_chat import message
from langchain_ollama import ChatOllama
from ingestion import vectorStore

load_dotenv()
embeddings=OllamaEmbeddings(model="mxbai-embed-large")
vectorStore=PineconeVectorStore(index_name="langchain-docs-2026",embedding=embeddings)
model = ChatOllama(model="qwen3:latest")
@tool(response_format="content_and_artifact",description="Retrieve relevant documentation for answering LangChain questions.")
def retrieve_content(query:str):
    print("Retriving")
    """Retrieve relevent documentation to help answer user queries about langchain"""
    retrieved_docs=vectorStore.as_retriever().invoke(query,k=4)
    # serialize documents for a model
    serialized="\n\n".join((f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}")
                           for doc in retrieved_docs
                           )
    return serialized, retrieved_docs
def run_llm(query:str)->Dict[str,Any]:
    """
        Run the RAG Pipeline to answer a query using retreived document
        Args:
            query: The user's question
        Returns:
            Dictionary containing:
                - answer: The generated answer
                - context: List of retreived documents
    """
    # create a agent with retreived tool
    system_prompt=(
        "You are a helpful AI assistant that answer questions about langchain documents"
        "You have access to a tool that retrives relevent information"
        "use the tool to find relevent information before answering questions"
        "Always cite the sources you use in your answers"
        "If you cannot find the answer in the retreived documentation say so"
    )
    agent=create_agent(model,tools=[retrieve_content],system_prompt=system_prompt)
    #Build messages list
    messages=[{"role":"user","content":query}]
    #Invoke the agent
    response=agent.invoke({"messages":messages})
    #Extract the answer from the last AI message
    answer=response["messages"][-1].content
    #Extract context document from Tool Message artifacts
    context_docs=[]
    for message in response["messages"]:
        #check if this is a tool message with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            #The artifact should contain the list of document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {
        "answer":answer,
        "context":context_docs
    }
if __name__=="__main__":
    result=run_llm(query="what are deep agents?")
    print(result)




