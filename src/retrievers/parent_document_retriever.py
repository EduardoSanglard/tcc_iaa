"""
Testing with the Parent Document Retrievel methodology
Available at
https://python.langchain.com/docs/how_to/parent_document_retriever/
"""

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data_path = "./data/Edital - Recortado - Optimized.pdf"

# Loading Documents
docs = []
document = PyPDFLoader(data_path).load()
docs.extend(document)

# Loading Documents
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=20,
    length_function=len,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OllamaEmbeddings(model="mistral-nemo:latest"),
)

# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)
print(list(store.yield_keys()))

sub_docs = vectorstore.similarity_search("Posso usar meu nome social")
print(sub_docs[0].page_content)

retrieved_docs = retriever.invoke("Como consigo insenção da taxa de inscrição")
len(retrieved_docs[0].page_content)
print(retrieved_docs[0].page_content)
