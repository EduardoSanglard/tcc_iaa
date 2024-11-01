import warnings

warnings.filterwarnings("ignore", message="Signature.*longdouble.*")

import erag, ast
import pandas as pd
from typing import Dict, List
from rouge_score import rouge_scorer


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


def text_generator(queries_and_documents: Dict[str, List[str]]) -> Dict[str, str]:

    results: Dict[str, str] = {}

    # Read pdf file with answers
    df = pd.read_excel(
        "./data/faqs/Respostas_FAQ_Completo_SemContexto.xlsx", sheet_name="FAQ_Manual"
    )

    # Build Expected Outputs with questions and answers columns
    for question, documents in queries_and_documents.items():
        print(f"Looking for question: {question}")
        df_question = df.loc[df["question"] == question]
        if df_question.empty:
            print(f"Question not found: {question}")
            continue
        results[question] = df_question["answer"].iloc[0]

    return results


def main():

    excel_model_faqs_path = "./data/faqs/answers_generated/FAQ_gemma2-9b.xlsx"
    df = pd.read_excel(excel_model_faqs_path, sheet_name="faq_with_answers")
    # retrieval_metrics = {'P_10', 'map'}
    retrieval_metrics = {"P_10", "P_3", "map"}

    print("Iniciado execução")

    print("Montando listas de perguntas, respostas e contextos por pergunta")
    # Build Expected Outputs with questions and answers columns
    expected_outputs = {}
    queries_and_documents = {}
    for index, row in df.iterrows():
        question = row["question"]
        print(f"Adicionando pergunta: {question}")
        expected_outputs[question] = [row["ground_truth"]]
        str_contexts = str(row[f"contexts"])
        str_contexts = str_contexts.replace("\n", "")
        str_contexts = str_contexts.replace("\r", "")
        str_contexts = str_contexts.replace("\t", "")
        str_contexts = str_contexts.replace("['", "").replace("']", "")
        list_contexts = str_contexts.split("','")
        queries_and_documents[question] = list_contexts

    tex_gen_function = text_generator

    results = erag.eval(
        retrieval_results=queries_and_documents,
        expected_outputs=expected_outputs,
        text_generator=tex_gen_function,
        downstream_metric=rouge_metric,
        retrieval_metrics=retrieval_metrics,
    )
    score_columns = ["question"]
    for metric in retrieval_metrics:
        score_columns.append(metric)

    df_metrics = pd.DataFrame(columns=score_columns)
    print(results)
    for query, result in results["per_input"].items():
        print(f"Query: {query} - Result: {result}")
        new_row = pd.DataFrame(
            [
                {
                    "question": query,
                    "P_10": result["P_10"],
                    "P_3": result["P_3"],
                    "map": result["map"],
                }
            ]
        )
        # new_row = pd.DataFrame([{'Model': "", 'P_3': result['P_3'], 'success': result['success']}])
        df_metrics = pd.concat([df_metrics, new_row], ignore_index=True)

    # Save DataFrame with all metrics
    df_metrics.to_csv("./data/scores/erag_gemma_metrics.csv", index=False)

    """
        
	#retrieval_metrics = {'P_10', 'success', 'P_3'}
	

	# List Models
	model ='Meta-Llama-3.1-70B-Instruct'

	print("Coletando métricas de respostas para o modelo Llama\n\n")

	df_metrics = pd.DataFrame(columns=['Model', 'P_3', 'success'])
        
    print(f"Coletando feedback de modelo: ")

        


    """


if __name__ == "__main__":
    main()
