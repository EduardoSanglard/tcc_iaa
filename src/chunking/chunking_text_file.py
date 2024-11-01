from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data_path = "data/edital_text/edital_text.txt"
chunk_size = 3000
chunk_overlap = 50

loaders = [TextLoader("data/edital_text/edital_text.txt")]
docs = []
for l in loaders:
    docs.extend(l.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_documents(docs)

print("Total Chunks: ", len(chunks))

for i, chunk in enumerate(chunks):
    print(chunk)
    print("\n\n")
    # Save Chunk in text file
    with open(f"data/chunks/chunk_{i+1}.txt", "w") as f:
        f.write(chunk.page_content)
