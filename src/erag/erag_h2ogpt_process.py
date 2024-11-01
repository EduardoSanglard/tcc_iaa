import warnings

warnings.filterwarnings("ignore", message="Signature.*longdouble.*")

import erag
import pandas as pd
from typing import Dict, List
from rouge_score import rouge_scorer


def extrair_lista_contextos(textoContextos: str) -> List[str]:

    # Show first 10 characters of string
    print(f"Extraindo contextos do texto bruto {textoContextos[:10]}")

    lista_contextos = []
    # Break text by line feed
    lines = textoContextos.split("\n")
    contexto_atual = ""
    for line in lines:

        if line == '"""':
            continue

        if "Document Contents" in line:
            contexto_atual = "{"
            continue

        if "End Document" in line:
            contexto_atual = contexto_atual[1:]
            lista_contextos.append(contexto_atual)
            contexto_atual = ""
            continue

        if contexto_atual != "":
            contexto_atual += line + "\n"
            continue

    return lista_contextos


def rouge_metric(generated_outputs, expected_outputs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = dict()
    for query, gen_output in generated_outputs.items():
        expe_outputs_query = expected_outputs[query]
        max_value = 0
        for exp_output in expe_outputs_query:
            max_value = max(
                scorer.score(exp_output, gen_output)["rougeL"].fmeasure, max_value
            )
        results[query] = max_value
    return results


def text_generator_llama(queries_and_documents: Dict[str, List[str]]) -> Dict[str, str]:

    results: Dict[str, str] = {}

    # Read pdf file with answers
    df = pd.read_excel("data/Respostas_FAQ_Completo.xlsx", sheet_name="Respostas")

    # Build Expected Outputs with questions and answers columns
    for question, documents in queries_and_documents.items():
        print(f"Looking for question: {question}")
        df_question = df.loc[df["question"] == question]
        if df_question.empty:
            print(f"Question not found: {question}")
            continue
        results[question] = df_question["Meta-Llama-3.1-70B-Instruct"].iloc[0]

    return results


def text_generator_gpt(queries_and_documents: Dict[str, List[str]]) -> Dict[str, str]:

    results: Dict[str, str] = {}

    # Read pdf file with answers
    df = pd.read_excel("data/Respostas_FAQ_Completo.xlsx", sheet_name="Respostas")

    # Build Expected Outputs with questions and answers columns
    for question, documents in queries_and_documents.items():
        print(f"Looking for question: {question}")
        df_question = df.loc[df["question"] == question]
        if df_question.empty:
            print(f"Question not found: {question}")
            continue
        results[question] = df_question["gpt-3.5turbo-0613"].iloc[0]

    return results


def text_generator_gemma(queries_and_documents: Dict[str, List[str]]) -> Dict[str, str]:

    results: Dict[str, str] = {}

    # Read pdf file with answers
    df = pd.read_excel("data/Respostas_FAQ_Completo.xlsx", sheet_name="Respostas")

    # Build Expected Outputs with questions and answers columns
    for question, documents in queries_and_documents.items():
        print(f"Looking for question: {question}")
        df_question = df.loc[df["question"] == question]
        if df_question.empty:
            print(f"Question not found: {question}")
            continue
        results[question] = df_question["google/gemma-2-27b-it"].iloc[0]

    return results


def text_generator_h2o(queries_and_documents: Dict[str, List[str]]) -> Dict[str, str]:

    results: Dict[str, str] = {}

    # Read pdf file with answers
    df = pd.read_excel("data/Respostas_FAQ_Completo.xlsx", sheet_name="Respostas")

    # Build Expected Outputs with questions and answers columns
    for question, documents in queries_and_documents.items():
        print(f"Looking for question: {question}")
        df_question = df.loc[df["question"] == question]
        if df_question.empty:
            print(f"Question not found: {question}")
            continue
        results[question] = df_question["h2oao/h2o-danube3-4b-chat"].iloc[0]

    return results


def main():
    print("Iniciado execução")

    print("Lendo Excel com perguntas e respostas")
    # Load Excel file with questions, contexts and answers
    df = pd.read_excel("data/Respostas_FAQ_Completo.xlsx", sheet_name="Respostas")

    # retrieval_metrics = {'P_10', 'success', 'P_3'}
    retrieval_metrics = {"P_3", "success"}

    # List Models
    models = {
        "Meta-Llama-3.1-70B-Instruct",
        "gpt-3.5turbo-0613",
        "google/gemma-2-27b-it",
        "h2oao/h2o-danube3-4b-chat",
    }

    print("Coletando métricas de respostas para o modelo Llama\n\n")

    df_metrics = pd.DataFrame(columns=["Model", "P_3", "success"])

    for model in models:

        print(f"Coletando feedback de modelo: " + model)

        print("Montando listas de perguntas, respostas e contextos por pergunta")
        # Build Expected Outputs with questions and answers columns
        expected_outputs = {}
        queries_and_documents = {}
        for index, row in df.iterrows():
            question = row["question"]
            print(f"Adicionando pergunta: {question}")
            expected_outputs[question] = [row["answer"]]
            queries_and_documents[question] = extrair_lista_contextos(
                row[f"context_{model}"]
            )

        if model == "Meta-Llama-3.1-70B-Instruct":
            tex_gen_function = text_generator_llama
        elif model == "gpt-3.5turbo-0613":
            tex_gen_function = text_generator_gpt
        elif model == "google/gemma-2-27b-it":
            tex_gen_function = text_generator_gemma
        elif model == "h2oao/h2o-danube3-4b-chat":
            tex_gen_function = text_generator_h2o

        results = erag.eval(
            retrieval_results=queries_and_documents,
            expected_outputs=expected_outputs,
            text_generator=tex_gen_function,
            downstream_metric=rouge_metric,
            retrieval_metrics=retrieval_metrics,
        )
        # print(results)
        for query, result in results["per_input"].items():
            print(f"Query: {query} - Result: {result}")
            new_row = pd.DataFrame(
                [{"Model": model, "P_3": result["P_3"], "success": result["success"]}]
            )
            df_metrics = pd.concat([df_metrics, new_row], ignore_index=True)

    # Save DataFrame with all metrics
    df_metrics.to_csv("data/metrics.csv", index=False)


if __name__ == "__main__":
    main()
