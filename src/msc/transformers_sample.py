import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import QATemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Set up the environment
os.environ["HF_TOKEN"] = "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"

# Load the Hugging Face model
tokenizer = AutoTokenizer.from_pretrained("h2oai/h2o-danube2-1.8b-base")
model = AutoModelForCausalLM.from_pretrained("h2oai/h2o-danube2-1.8b-base")

# Create a Hugging Face pipeline
llm = HuggingFacePipeline(model=model, tokenizer=tokenizer)

# Load the knowledge base
documents = TextLoader("path/to/knowledge_base.txt").load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store
vectorstore = FAISS.from_embeddings(embeddings.embed_documents(docs))

# Create a QA template
qa_template = QATemplate(input_variables=["query", "context"])

# Create the RAG chain
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    qa_template=qa_template,
)

# Test the RAG pipeline
query = "What is the capital of France?"
result = retrieval_qa_chain.run(query=query)
print(result)
