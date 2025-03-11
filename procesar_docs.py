import os
from dotenv import load_dotenv
import pandas as pd
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Cargar variables de entorno
load_dotenv()

# Función para leer archivos Excel
def load_excel(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    text = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    return text

# Función para leer archivos PDF
def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            try:
                text += page.extract_text() or ""
            except Exception as e:
                print(f"⚠️ Error al leer página en {file_path}: {e}")
    return text

# Lista de documentos
documentos = [
    "docs/mvll.pdf",
    #"docs/manual_usuario.pdf"
]

# Leer documentos y almacenarlos en una lista
text_list = []
for doc in documentos:
    if doc.endswith(".xlsx"):
        text_list.append(load_excel(doc))
    elif doc.endswith(".pdf"):
        text_list.append(load_pdf(doc))

# Unir textos
all_text = "\n".join(text_list)

# Dividir en fragmentos y generar embeddings
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(all_text)

# Verificar si la API Key está configurada
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("🚨 ERROR: No se encontró la API Key de OpenAI. Asegúrate de configurarla en el archivo .env.")

# Generar embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# Guardar la base de conocimientos
knowledge_base.save_local("vectorstore")

print("✅ Base de conocimientos guardada correctamente.")
