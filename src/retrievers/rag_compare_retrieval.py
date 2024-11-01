from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

model_name = "mistral-nemo:latest"

data_path = "./data/Edital - Recortado - Optimized.pdf"
collection_name = "chroma_" + model_name.replace(":", "-")
vector_store_path = "./data/vector_store_chroma/" + collection_name

print("Creating embeddings and vector database...")
embedding_func = OllamaEmbeddings(
    model=model_name, show_progress=False, temperature=0.9
)

# Create a vectorstore
vectordb = Chroma(
    collection_name=collection_name,
    persist_directory=vector_store_path,
    embedding_function=embedding_func,
)


# Create different retriever strategies and compare each strategy's performance
retrievers = (
    vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    vectordb.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3}
    ),
)


while True:
    # Perform a query
    query = input("Ask a question (or type 'quit' to exit): ")

    if query == "quit":
        exit()

    for i, retriever in enumerate(retrievers):
        print("Retriever: ", i)
        docs = retriever.invoke(query)
        for doc in docs:
            print(doc)
