""" Script to read the full answers from the FAQ file, split the list of contexts and generate a row for each context"""

import pandas as pd
from typing import Dict, List
import ast

excel_file_path = "./data/faqs/FAQ - Contextos.xlsx"

df = pd.read_excel(excel_file_path, sheet_name="Contextos")

df_contexts = pd.DataFrame(columns=["# Pergunta", "Pergunta", "# Contexto", "Contexto"])

for index, row in df.iterrows():

    str_contexts = str(row["Lista Contextos"])
    str_contexts = str_contexts.replace("\n", " ")
    str_contexts = str_contexts.replace("\r", " ")
    str_contexts = str_contexts.replace("\t", " ")
    str_contexts = str_contexts.replace("['", "").replace("']", "")

    contextos = str_contexts.split("','")

    for idx, ctx in enumerate(contextos):
        print(f"\n\nAdicionando contexto: {ctx}")

        df_contexts = df_contexts._append(
            {
                "# Pergunta": row["#"],
                "Pergunta": row["Pergunta"],
                "# Contexto": idx + 1,
                "Contexto": ctx,
            },
            ignore_index=True,
        )

df_contexts.to_excel("./data/faqs/FAQ - Contextos - Split.xlsx", index=False)

print(df.columns)
