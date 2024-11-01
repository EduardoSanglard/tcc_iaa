from datasets import Dataset
import json
import pandas as pd
import os

data_path = "data/openwebui_chat.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

format_data = {}
questions = []
answers = []
contexts = []

for i, chat in enumerate(data[0]["chat"]["messages"]):
    # print(chat)

    if chat["role"] == "user":
        print(f"Question: {chat['content']}")
        questions.append(chat["content"])
    elif chat["role"] == "assistant":
        print(f"Answer: {chat['content']}")
        answers.append(chat["content"])

    # Check if contains citations
    if "citations" in chat:
        doc_citations = chat["citations"][0]["document"]
        # Remove all line breaks from the text
        for i, doc in enumerate(doc_citations):
            doc_citations[i] = doc.replace("\n", " ")
        contexts.append(doc_citations)


data = {"questions": questions, "answers": answers, "contexts": contexts}

dataset = Dataset.from_dict(data)
df_dataset = pd.DataFrame(dataset)

# Add a 'ground_truth' column to the dataset
df_dataset["ground_truth"] = df_dataset["answers"]

# Read the Full FAQs dataset
df_faq = pd.read_excel(
    "data/faqs/Respostas_FAQ_Completo_ComContexto.xlsx", sheet_name="Repostas"
)

# Join both datasets, get the 'answer' value from the dt_faq and add to the ground_truth column based on the question
for i, row in df_dataset.iterrows():
    question = row["questions"]
    answer = df_faq[df_faq["question"] == question]["answer"].values
    if len(answer) > 0:
        df_dataset.at[i, "ground_truth"] = answer[0]

# if old file exists, remove it
if os.path.exists("data/openwebui_chat_dataset.csv"):
    os.remove("data/openwebui_chat_dataset.csv")

df_dataset.to_csv("data/openwebui_chat_dataset.csv", index=False, encoding="utf-8")
