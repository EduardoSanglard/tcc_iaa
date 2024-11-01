"""
Create different vector databases with different chunk sizes for the same embeddings.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import time

# Set execution start time
start_time = time.time()
print("Starting execution at %s" % time.ctime())

models = ["gemma2:9b", "mistral-nemo:latest", "llama3.1:8b"]
chunk_sizes = [1000, 2000, 3000, 4000, 5000]
data_path = "./data/Edital - Recortado - Optimized.pdf"

for model in models:
    print("Creating Vector DBs for Model: " + model)

    embedding_func = OllamaEmbeddings(model=model, show_progress=False, temperature=0.5)

    vector_store_path = "./data/vector_store_chroma/" + model.replace(":", "-")

    for chunk_size in chunk_sizes:

        print("Creating Vector DB for Chunk Size: " + str(chunk_size))

        collection_name = f"chroma_{chunk_size}_" + model.replace(":", "-")

        print("Creating Collection: " + collection_name)
        print("Loading PDF and splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
        )
        documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)
        # Show total amount of tokens in the collection
        print(f"Total chunks {len(documents)} chunks")

        vectordb = Chroma.from_documents(
            documents,
            embedding=embedding_func,
            collection_name=collection_name,
            persist_directory=vector_store_path,
        )

        print("Vector DB created for Chunk Size: " + str(chunk_size))


# Print the execution time in minutes and seconds
print(
    "Execution time: %s minutes and %s seconds" % divmod((time.time() - start_time), 60)
)
