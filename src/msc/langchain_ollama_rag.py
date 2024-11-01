from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import os
import pandas as pd
import time

# Set execution start time
start_time = time.time()

model_name = "llama3.1:8b"
model_name = "mistral-nemo:latest"
model_name = "gemma2:9b"

print("Starting execution at %s" % time.ctime())

data_path = "./data/edital_text/edital_text.txt"
collection_name = "chroma_" + model_name.replace(":", "-")
vector_store_path = "./data/vector_store_chroma/" + collection_name

print("Creating embeddings and vector database...")
embedding_func = OllamaEmbeddings(
    model=model_name, show_progress=False, temperature=0.3
)

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
    print("Loading PDF and splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    documents = TextLoader(data_path).load_and_split(text_splitter=text_splitter)
    vectordb = Chroma.from_documents(
        documents,
        embedding=embedding_func,
        collection_name=collection_name,
        persist_directory=vector_store_path,
    )

# Show total amount of tokens in the collection
print(
    f"Total tokens in collection: {vectordb.total_tokens} separeted into {vectordb.total_chunks} chunks"
)

print("Forming template and creating RAG chain...")
# Read the template from a text file
with open("data/template_prompt.txt", "r") as file:
    template = file.read()

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(model=model_name, max_tokens=206, temperature=0.3, top_k=75, top_p=0.2)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

print("RAG Chain Startup Complete.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain)


# Function to ask questions
def ask_question(question):

    result = rag_chain_source.invoke({"question": question})
    print("Answer:\n\n " + result["answer"])
    print("\n Contexts: ")
    for i, doc in enumerate(result["context"]):
        print(f"Doc {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print("---")


# Example usage
if __name__ == "__main__":

    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == "quit":
            break
        answer = ask_question(user_question)
        print("\n")

    # Print the execution time in minutes and seconds
    print(
        "Execution time: %s minutes and %s seconds"
        % divmod((time.time() - start_time), 60)
    )
