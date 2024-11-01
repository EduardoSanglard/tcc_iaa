from langchain.callbacks.manager import CallbackManager
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class RetrievalLogger:
    def __init__(self):
        self.retrieved_docs = []

    def retrieve_callback(self, retrieved_docs):
        print("Retrieved Documents:")
        self.retrieved_docs.extend(retrieved_docs)


model_name = "h2oai/h2o-danube2-1.8b-base"

data_path = "./data/Edital - Recortado - Optimized.pdf"
collection_name = "chroma_" + model_name.replace(":", "-")
vector_store_path = "./data/vector_store_chroma/" + collection_name

print("Creating embeddings and vector database...")
embedding_func = HuggingFaceEmbeddings(model_name=model_name)

# Create a vectorstore
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=vector_store_path,
    embedding_function=embedding_func,
)

# Initialize our logger
logger = RetrievalLogger()


"""
Create different retriever strategies and compare each strategy's performance



"""

# Create a retriever with the callback
retriever = vectorstore.as_retriever(
    callback_manager=CallbackManager([logger.retrieve_callback])
)


while True:
    # Perform a query
    query = input("Ask a question (or type 'quit' to exit): ")

    if query == "quit":
        exit()

    docs = retriever.invoke(query)
    for doc in docs:
        print(doc)
