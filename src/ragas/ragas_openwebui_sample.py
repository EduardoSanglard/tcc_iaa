from datasets import Dataset
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Get the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

df_answer = pd.read_csv("data/openwebui_chat_dataset.csv")
dataset = Dataset.from_pandas(df_answer)

# Set the 'contexts' columns to be a list
dataset = dataset.map(lambda x: {"contexts": [x["contexts"]]})

score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
score = score.to_pandas()
score.to_csv("data/score_openwebui_gpt4o.csv")
