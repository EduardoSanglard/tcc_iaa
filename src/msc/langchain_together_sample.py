from langchain_together import Together
from langchain_together import TogetherEmbeddings

together_key = "acd6c23979b489fb565b4736e001ee4019b9d7c3424b285e6aba6ec7e3030c26"

embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

together_completion = Together(
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    temperature=0.7,
    max_tokens=4000,
    top_k=1,
    together_api_key=together_key,
)

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from ragas import evaluate
from ragas.metrics import faithfullness, fluency, relevance
from datasets import Dataset

langchain_llm = ChatOllama(model="mistral-nemo:latest", max_tokens=128, temperature=0.3)

langchain_embeddings = OllamaEmbeddings(
    model="mistral-nemo:latest", show_progress=False
)

data_samples = {
    "question": ["When was the first super bowl?", "Who won the most super bowls?"],
    "answer": [
        "The first superbowl was held on Jan 15, 1967",
        "The most super bowls have been won by The New England Patriots",
    ],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
        ],
        [
            "The Green Bay Packers...Green Bay, Wisconsin.",
            "The Packers compete...Football Conference",
        ],
    ],
    "ground_truth": [
        "The first s*uperbowl was held on January 15, 1967",
        "The New England Patriots have won the Super Bowl a record six times",
    ],
}

dataset = Dataset.from_dict(data_samples)


results = evaluate(
    metrics=[faithfullness, fluency, relevance],
    llm=langchain_llm,
    embeddings=embeddings,
)
print(results)
