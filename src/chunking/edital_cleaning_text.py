import os, re

edital_text = "data/Edital - Recortado - Optimized.txt"

# Read PDF file
with open(edital_text, "r", encoding="utf-8") as file:
    edital_text = file.read()
    file.close()

# Strip all lines of text
edital_text = "\n".join([line.strip() for line in edital_text.split("\n")])

# Remove all occurrences of text 'PS-2024 – Edital n° 24/2023 – Página x de 82' with regex
edital_text = re.sub(r"PS-2024 – Edital n° 24/2023 – Página \d+ de 82", "", edital_text)

# Remover erratas
edital_text = re.sub(r"\d+ Alterado pelo .* de 202(3|4)[.]", "", edital_text)

# Replace unreadable characters for - character
edital_text = edital_text.replace("�", "- ")
edital_text = edital_text.replace("", "- ")

# Replace two or more line break with only two
edital_text = re.sub(r"\n{2,}", "\n\n", edital_text)

inicio_cronograma = False
linha_cronograma = ""
linhas_cronograma = []
linhas_remover = []

for i, line in enumerate(edital_text.split("\n")):

    if "Anexo I – Cronograma4546474849" == line:
        inicio_cronograma = True
        print("Incício do cronograma")

    if not inicio_cronograma:
        continue

    linhas_cronograma.append(line)

    # If line match with Regex r"^\d+[.]\d+
    if re.match(r"^\d+[.]\d+", line):
        if linha_cronograma:
            linhas_cronograma.append(linha_cronograma)
            linha_cronograma = ""
        # print(line)
        linha_cronograma = line
    else:
        linha_cronograma += " " + line

    # if inicio_cronograma:
    #    print(line)

    if "Anexo II – Comprovação De Renda60" == line:
        inicio_cronograma = False
        print("Fim do cronograma")
        break

# Salvar tabela cronograma em arquivo separado
cronograma_file = "data/edital_text/cronograma.txt"
if os.path.exists(cronograma_file):
    os.remove(cronograma_file)
with open(cronograma_file, "w", encoding="utf-8") as file:
    for line in linhas_cronograma:
        file.write(line + "\n")
    file.close()

# Save text in a file. Delete old if exists
cleaned_text_file = "data/edital_text/edital_text.txt"

if os.path.exists(cleaned_text_file):
    os.remove(cleaned_text_file)
with open(cleaned_text_file, "w", encoding="utf-8") as file:
    file.write(edital_text)
    file.close()
