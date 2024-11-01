from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# My list of Ollama Models
models_list = ["gemma2:9b", "llama3.1:8b", "mistral-nemo:latest"]
models_list = ["mistral-nemo:latest"]

# loader = PyPDFLoader("data/Edital n.º 24_2023-NC_PROGRAD - Processo Seletivo UFPR 2024  (compilado com retificações).pdf")
# pages = loader.load_and_split()

for model in models_list:

    print(f"Testing with model: {model}")

    llm = ChatOllama(model=model, temperature=0.3)

    embeddings = OllamaEmbeddings()

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
