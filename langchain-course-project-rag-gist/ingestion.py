import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()




if __name__ == "__main__":
    print("Ingesting...")
    loader=TextLoader("mediumblog1.txt",encoding = 'UTF-8')
    document=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
    PineconeVectorStore.from_documents(texts, embeddings,index_name=os.environ.get("INDEX_NAME"))

    print("finish")