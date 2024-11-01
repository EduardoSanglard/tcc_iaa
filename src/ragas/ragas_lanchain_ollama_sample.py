from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision
from datasets import Dataset

from ragas import RunConfig

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

langchain_llm = ChatOllama(model="mistral-nemo:latest", max_tokens=128, temperature=0.9)

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

run_config = RunConfig(timeout=720)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_correctness, context_precision],
    llm=langchain_llm,
    embeddings=langchain_embeddings,
    run_config=run_config,
)
score = results.to_pandas()
print(score)

score.to_csv("data/score_ollama_mistral.csv")
