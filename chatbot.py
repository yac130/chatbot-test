import streamlit as st
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


# Cargar la base de conocimientos guardada
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.load_local("vectorstore", embeddings)

# Configurar la app en Streamlit
st.title("🤖 Chatbot con Base de Conocimientos")
st.write("Haz preguntas sobre los documentos preprocesados.")

# Campo para hacer preguntas
pregunta = st.text_input("✍️ Escribe tu pregunta:")

if pregunta:
    retriever = knowledge_base.as_retriever()
    similar_docs = retriever.get_relevant_documents(pregunta)
    context = "\n".join([doc.page_content for doc in similar_docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    respuesta = llm.predict(f"Usa la siguiente información para responder:\n\n{context}\n\nPregunta: {pregunta}")

    st.subheader("🤔 Respuesta:")
    st.write(respuesta)
