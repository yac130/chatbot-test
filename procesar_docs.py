import os
from dotenv import load_dotenv
import pandas as pd
import pypdf
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


# Función para leer archivos Excel
def load_excel(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    text = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    return text

# Función para leer archivos PDF
def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Cargar y procesar archivos (ajusta las rutas)
documentos = [
    "docs/mvll.pdf"
    #"docs/manual_usuario.pdf"
]

all_text = ""
for doc in documentos:
    if doc.endswith(".xlsx"):
        all_text += load_excel(doc)
    elif doc.endswith(".pdf"):
        all_text += load_pdf(doc)

# Dividir en fragmentos y generar embeddings
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(all_text)
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# Guardar la base de conocimientos
knowledge_base.save_local("vectorstore")

print("✅ Base de conocimientos guardada correctamente.")
