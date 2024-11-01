"""
Compare a single questions with 3 different embedding models, their databases collections with different chunk sizes.
Save all print statements in a log file.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

import time, os
import logging

data_path = "./data/Edital - Recortado - Optimized.pdf"

# Configure basic logging settings
logging.basicConfig(
    filename="data/rag_outputs/rag_testing.log",  # Name of the log file
    level=logging.INFO,  # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Custom log format
    filemode="a",  # Append mode ('w' for overwrite)
)

query = input("Enter a question: ")

# Set execution start time
start_time = time.time()
logging.info("Starting execution at %s" % time.ctime())


models = ["phi3.5"]
models = ["gemma2:9b", "mistral-nemo:latest", "llama3.1:8b"]
# chunk_sizes = [1000, 2000, 3000, 4000, 5000]
chunk_sizes = [3000]

for model in models:

    logging.info("\n\nCreating Embeddings for Model: " + model)

    embedding_func = OllamaEmbeddings(model=model, show_progress=False, temperature=0.3)

    vector_store_path = "./data/vector_store_chroma/" + model.replace(":", "-")

    for chunk_size in chunk_sizes:

        # Loading Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
        )
        documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)

        # Add Metadata to the documents
        for i, doc in enumerate(documents):
            doc.metadata = {"id": i}

        collection_name = f"chroma_{chunk_size}_" + model.replace(":", "-")

        logging.info("Loading Collection already created: " + collection_name)

        # if vector store path already exists, load it
        if os.path.exists(vector_store_path):
            print("Loading Collection already created: " + collection_name)
            vectordb = Chroma(
                collection_name=collection_name,
                persist_directory=vector_store_path,
                embedding_function=embedding_func,
            )
        else:
            print("Creating Collection: " + collection_name)

            # Show total amount of tokens in the collection
            print(f"Total chunks {len(documents)} chunks")

            vectordb = Chroma.from_documents(
                documents,
                embedding=embedding_func,
                collection_name=collection_name,
                persist_directory=vector_store_path,
            )

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 2

        logging.info("Retrieving RAG results for question: " + query + "\n\n")

        chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )

        docs = retriever.invoke(query)

        # Save retrieved documents in a separated .txt file
        with open(f"data/rag_outputs/{model}_{chunk_size}_rag_results.txt", "w") as f:
            f.write(
                f"Retrieved Documents for Model: {model} and Chunk Size: {chunk_size}\n\n"
            )
            for doc in docs:
                f.write(str(doc) + "\n\n")


# Print the execution time in minutes and seconds
logging.info(
    "\n\nExecution time: %s minutes and %s seconds"
    % divmod((time.time() - start_time), 60)
)
