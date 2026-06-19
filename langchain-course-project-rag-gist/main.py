import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

print("Initializing components.....")
embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
llm=ChatOllama(model="gemma3:1b")
vectorStore=PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=
                                embeddings)
retriever=vectorStore.as_retriever(search_kwargs={"k":3})
prompt_template=ChatPromptTemplate.from_template(""" 
 Answer the question based only on the following context
 {context}
 Question:{question}
 provide a detailed answer
""")

def format_docs(docs):
    """Format retrived douments in to a single string"""
    return "\n\n".join(doc.page_content for doc in docs)
def create_retrivel_chain_with_lcel():
    """
        Create a retreival chain with LCEL( Langchain expression language)
        Return a chain that can be invoked with Pquestion:...}

        Advantages over non lcel
        - Declarative and Composable: Easy to chain operations with pipe operator (|)
        - Built in streaming: chain.stream() works out of the box
        - Built in async: chain.ainvoke() and chain.astream() available
        - Batch Processing: chain.batch() for multiple inputs
        - Type safety: Better integration with Langchain's type system
        - Less code: More consice and readeable
        - Reusable: chain can be saved, shared, and composed with other chains
        - Better debugging: langchain provides better observability tools
    """
    retrievel_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question")  | retriever | format_docs
        ) | prompt_template | llm | StrOutputParser()

    )
    return retrievel_chain

# ================================================================
# with out lcel (function based approch)
# ================================================================
def retrievel_chain_without_lcel(query: str):
    """
        Simple retrievel chain withput lcel
        manually retreives document, formats them, and generate a response
        Limitations:
        - manual step by step execution
        - No built in streaming support
        - No async support with out additional code
        - Harder to compose with other chains
        - More verbose and error prone
    """
    docs=retriever.invoke(query)
    context=format_docs(docs)
    messages=prompt_template.format_messages(context=context,question=query)
    response=llm.invoke(messages)
    return response.content

if __name__=="__main__":
    print("Retrieving....")
    query="what is pinecoin in machine learning?"
    result_without_lcel=retrievel_chain_without_lcel(query)
    print(result_without_lcel)
    print("===========================================================")
    chain_with_lcel=create_retrivel_chain_with_lcel()
    result_with_lcel= chain_with_lcel.invoke({"question":query})
    print("\nAnswer")
    print(result_with_lcel)
