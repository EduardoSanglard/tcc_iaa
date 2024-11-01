"""
Version of the main execution workflow, but designed to be used with custom prompts

"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain.prompts import ChatPromptTemplate

from langchain_community.chat_models import ChatOllama

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

import os
import time
import pandas as pd
from typing import List
from dotenv import load_dotenv


# Set execution start time
start_time = time.time()
print("Starting execution at %s" % time.ctime())

load_dotenv()

# Get the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

model_name = "openai"
model_name = "mistral-nemo"
model_name = "llama3.1:8b"
model_name = "gemma2:9b"
chunk_size = 1000

data_path = "./data/edital_text/edital_text.txt"
# collection_name = f"chroma_{chunk_size}_"+model_name.replace(":", "-")
# vector_store_path = "./data/vector_store_chroma/"+model_name.replace(":", "-")

collection_name = "bge-m3-1000"
vector_store_path = "./data/vector_store_chroma/bge-m3"


print("Loading Source Text")
loaders = [TextLoader(data_path)]
docs = []
for l in loaders:
    docs.extend(l.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=50, length_function=len
)
documents = text_splitter.split_documents(docs)


print("Creating embeddings")
if model_name == "openai":
    embedding_func = OpenAIEmbeddings()
else:
    # embedding_func = OllamaEmbeddings(model=model_name)
    embedding_func = OllamaEmbeddings(model="bge-m3")

# if vector store path already exists, load it
if os.path.exists(vector_store_path):
    # if 1 < 0:
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

print("Forming template and creating RAG chain...")
# Read the template from a text file
with open("data/template_prompt.txt", "r") as file:
    template = file.read()

prompt = ChatPromptTemplate.from_template(template)

if model_name == "openai":
    llm = ChatOpenAI(model="gpt-4o", max_tokens=206, temperature=0.3, top_p=0.2)
else:
    llm = ChatOllama(
        model=model_name, max_tokens=206, temperature=0.3, top_k=75, top_p=0.2
    )


chroma_retriever = vectordb.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)


def bm25_preprocessor(text) -> List[str]:
    if isinstance(text, dict):
        return [str(text["question"])]
    return text.split()


bm25_retriever = BM25Retriever.from_documents(
    documents, preprocess_func=bm25_preprocessor
)
bm25_retriever.k = 2

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.3, 0.7]
)

print("RAG Chain Startup Complete.")


def format_docs(docs) -> str:

    return str(docs)

    # if docs is a string
    if isinstance(docs, str):
        return docs
    return "\n\n".join(doc.page_content for doc in docs)


"""
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain_source = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

"""

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": format_docs | retriever, "question": RunnablePassthrough()}
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

    excel_faqs_path = "data/faqs/Respostas_FAQ_Completo_SemContexto.xlsx"
    excel_model_faqs_path = (
        "data/faqs/answers_generated/FAQ_" + model_name.replace(":", "-") + ".xlsx"
    )

    # Read FAQs data
    df_faqs = pd.read_excel(excel_faqs_path, sheet_name="faq")

    # Rename the 'answer' column to 'ground_truth'
    df_faqs.rename(columns={"answer": "ground_truth"}, inplace=True)

    # Add an 'answer' and 'context' column to the FAQs dataframe
    df_faqs["answer"] = ""
    df_faqs["contexts"] = ""

    while True:
        question = input("Faça uma pergunta ao ChatBot NC: ")
        if question == "exit":
            break

        result = rag_chain_source.invoke(question)
        answer = result["answer"]

        print("Respostas:\n\n " + answer)

        contexts = []
        print("\n Contexts: ")
        for i, doc in enumerate(result["context"]):
            print(f"Doc nº {i+1}:")
            print(f"Conteúdo: {doc.page_content[:100]}...")
            contexts.append(doc.page_content)
            print(
                "\n-----------------------------------------------------------------------------------------------------------\n"
            )

    # Print the execution time in minutes and seconds
    print(
        "Execution time: %s minutes and %s seconds"
        % divmod((time.time() - start_time), 60)
    )
