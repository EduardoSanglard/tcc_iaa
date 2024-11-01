"""
    Testes realizados com a leitura e formatação do arquivo Edital em PDF
    Foi identificado quebra de palavras e textos desnecessárias na formatação do texto
    Este problema é inerente a biblioteca pdfReader, a conversão manual do texto não mostrou grandes problemas
    O experimento irá seguir de forma que trabalhe com o texto extraído e não com o texto original
"""

from pypdf import PdfReader
import re

data_path = "./data/Edital - Recortado - Optimized.pdf"
reader = PdfReader(data_path)
number_of_pages = len(reader.pages)
page = reader.pages[0]
full_text = ""


# Save Page text in a file
for i in range(number_of_pages):
    page = reader.pages[i]
    with open(f"data/text_pages/page_text{i}.txt", "w", encoding="utf-8") as file:
        page_text = page.extract_text()

        # Remove all empty lines from string
        # page_text = "\n".join([line for line in page_text.split("\n") if line.strip()])

        # Strip all lines of text
        page_text = "\n".join([line.strip() for line in page_text.split("\n")])

        # Remove all occurrences of text 'PS-2024 – Edital n° 24/2023 – Página x de 82' with regex
        page_text = re.sub(
            r"PS-2024 – Edital n° 24/2023 – Página \d+ de 82", "", page_text
        )

        # Remove ocorrencias de 2 Alterado pelo Edital de Retificação nº 13/2023-NC/PROGAD de 31 de maio de 2023.
        page_text = re.sub(
            r"\d+ Alterado pelo Edital de Retificação nº \d+/\d+-NC/PROGAD de \d+ de \w+ de \d+.",
            "",
            page_text,
        )

        # Replace unreadable characters for - character
        page_text = page_text.replace("�", "- ")
        page_text = page_text.replace("", "- ")

        full_text += page_text
        file.write(page_text)
        file.close()

with open(f"data/edital_text/edital_text.txt", "w", encoding="utf-8") as file:
    file.write(full_text)
    file.close()
