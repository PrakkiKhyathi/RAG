import asyncio
import os
import ssl
from typing import Any,List,Dict
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap
from openai import batches
from sqlalchemy.dialects.oracle.dictionary import all_synonyms
from sqlalchemy.testing.suite.test_reflection import metadata
from streamlit import success

from logger import (Colors,log_error,log_info,log_success, log_warning,log_header)

load_dotenv()
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()
print(OllamaEmbeddings.model_fields.keys())
embeddings=OllamaEmbeddings(model="mxbai-embed-large")
print(OllamaEmbeddings.model_fields.keys())
vectorStore=PineconeVectorStore(index_name="langchain-docs-2026",embedding=embeddings)
tavily_extract=TavilyExtract()
tavily_map=TavilyMap(max_depth=5,max_breadth=20,max_pages=1000)
tavily_crawl=TavilyCrawl()

async def main():
    """ Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info("TavilyCrawl: Starting to crawl documentation from http://python.langchain.com",Colors.PURPLE)
    # res=tavily_crawl.invoke({"url": "https://python.langchain.com","max_depth":1,"extract_depth":"advanced"})
    # all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in
    #             res["results"]]
    # log_success(f"TavilyCrawl: Sucessfully Crawled {len(all_docs)} URLS from documentation site")

    site_map=tavily_map.invoke("https://python.langchain.com/")
    log_success(f"TavilyCrawl: Successfully Crawled {len(site_map['results'])} URLS from documentation site")
    # split URLS in to batches of 20
    url_batches=chunk_urls(list(site_map["results"]),chunk_size=20)
    log_info(f"URL Processing split {len(site_map['results'])} URLS in to {len(url_batches)} batches",Colors.BLUE)
    all_docs=await async_extract(url_batches)
    #split documents in to chunks
    log_header("Document chunking phases")
    log_info(f"Text splitter: Process: {len(all_docs)} documents with 4000 chunk size and 20 overlap",Colors.YELLOW)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
    splitted_docs=text_splitter.split_documents(all_docs)
    log_success(f"Text splitter: created {len(splitted_docs)} chunks from {len(all_docs)} documents")
    await index_documents_async(splitted_docs,batch_size=500)

async def extract_batch(urls:List[str],batch_num:int)-> List[Dict[str,Any]]:
    """Extract documents from a batch of urls"""
    try:
        log_info(f"TavilyExtract Processing Batch {batch_num} with {len(urls)} URLS",Colors.BLUE)
        docs=await tavily_extract.ainvoke(input={"urls":urls})
        log_success(f"TavilyExtract: completed batch {batch_num} - extracted {len(docs.get('results',[]))} documents")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to {batch_num}-{e}")
        return []
async def async_extract(urls_batches: List[List[str]]):
    log_header("Document Extraction phase")
    log_info(f"Tavily Extract: starting concurrent extractions of {len(urls_batches)} batches",Colors.DARKCYAN)
    tasks=[extract_batch(batch,i+1) for i, batch in enumerate(urls_batches)]
    results=await asyncio.gather(*tasks,return_exceptions=True)
    all_pages=[]
    failed_batches=0
    for result in results:
        if isinstance(result,Exception):
            log_error(f"Tavily Extract: Batch Failed with Exception {result["results"]}")
            failed_batches+=1
        else:
            for extracted_page in result["results"]:
                document=Document(page_content=extracted_page["raw_content"],metadata={"source":extracted_page["url"]})
                all_pages.append(document)
    log_success(f"Tavily Extract: Extraction complete! Total pages extracted: {len(all_pages)}")
    if failed_batches>0:
        log_warning(f"Tavily Extract: {failed_batches} batches failed during extraction")
    return all_pages



def chunk_urls(urls: List[str], chunk_size:int =20) -> List[list[str]]:
    """ Split urls in to chunks of specified size"""
    chunks=[]
    for i in range(0,len(urls),chunk_size):
        chunk=urls[i: i+chunk_size]
        chunks.append(chunk)
    return chunks
async def index_documents_async(documents:List[Document],batch_size:int =50):
    """Process documents in batches asynhronously"""
    log_header("Vector storage phase")
    log_info(f"VectorStore Indexing: preparing to add {len(documents)} documents to vector store")
    batches=[documents[i:i+batch_size] for i in range(0,len(documents),batch_size)]
    log_info(f"Vector store indexing: split into {len(batches)} batches of {batch_size} document each")

    async def add_batch(batch: List[Document], batch_num:int):
        try:
            await vectorStore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} {len(batch)} Documents")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batches {batch_num}-{e}")
            return False
        return True

    tasks=[add_batch(batch,i+1) for i, batch in enumerate(batches)]
    results=await asyncio.gather(*tasks,return_exceptions=True)
    #count successful batches
    successful=sum(1 for result in results if result is True)
    if successful==len(batches):
        log_success(f"Vector Store Indexing All Batches processed successfully : ({successful}/{len(batches)})")
    else:
        log_warning(f"VectoreStore Indexing: Proccssed {successful}/{len(batches)} batches successfully")





if __name__=="__main__":
    asyncio.run(main())
