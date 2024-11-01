"""
This scripts contains the functions to clean the data from the edital.
The cleaned document list will be saved into a vector database for checking the quality of the retriever.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

import time
import logging


data_path = "./data/Edital - Recortado - Optimized.pdf"
chunk_size = 3000
chunks_folder = "./data/chunks"

# Loading Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=20,
    length_function=len,
)

loader = PyPDFLoader(data_path)
documents = loader.load_and_split(text_splitter=text_splitter)


# Save each chunk text
for i, doc in enumerate(documents):
    with open(f"{chunks_folder}/chunk_{i}.txt", "w") as file:
        file.write(doc.page_content)
        file.close()
