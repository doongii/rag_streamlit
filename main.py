import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ------------------------------------------------------------------------
# 1. Configuración inicial de Streamlit
# ------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chat - Parámetros Fijos", layout="centered")

# ------------------------------------------------------------------------
# 2. Definir la función que construye la cadena RAG con parámetros fijos
# ------------------------------------------------------------------------
def build_rag_chain(pdf_path, google_api_key):
    """
    Construye y retorna un RetrievalQA Chain con parámetros fijos.
    Modifica aquí si deseas cambiar la config de chunk_size, temperature, etc.
    """
    # A) Definir parámetros fijos
    TEMPERATURE = 0.05
    TOP_P = 0.95
    CHUNK_SIZE = 4096
    CHUNK_OVERLAP = 410
    K = 5

    # B) Configurar el modelo con parámetros fijos
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )

    # C) Cargar PDF y dividir en chunks
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # D) Embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    # E) Crear / cargar índice Chroma
    persist_directory = "./chroma_index"
    if os.path.exists(persist_directory):
        vector_index = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        ).as_retriever(search_kwargs={"k": K})
    else:
        vector_index = Chroma.from_texts(
            texts,
            embeddings,
            persist_directory=persist_directory
        ).as_retriever(search_kwargs={"k": K})

    # F) PromptTemplate personalizado
    template = """
    Usa las siguientes piezas de contexto para responder a la pregunta a continuación.
    Proporciona una respuesta lo más detallada posible, con explicaciones paso a paso
    y ejemplos relevantes. Si no sabes la respuesta, di que no la sabes.

    {context}

    Pregunta: {question}
    Respuesta detallada:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # G) Crear la cadena de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print(qa_chain)
    return qa_chain

# ------------------------------------------------------------------------
# 3. Interfaz principal con estilo chat
# ------------------------------------------------------------------------
def main():
    st.title("RAG Chat con Parámetros Fijos")

    # Verificar que tienes la API Key (aquí está hardcodeado,
    # si prefieres, usa os.getenv("GOOGLE_API_KEY"))
    GOOGLE_API_KEY = "AIzaSyDZz1R_g7QZD4bzNi_nX9yh24M3LbZx_hY"
    if not GOOGLE_API_KEY:
        st.error("⚠️ No se encontró la clave GOOGLE_API_KEY.")
        return

    pdf_path = "Documento PRUEBA - Encuesta.com.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"⚠️ No se encontró el PDF en la ruta: {pdf_path}")
        return

    # Cargar/crear la cadena de QA
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = build_rag_chain(pdf_path, GOOGLE_API_KEY)

    # Inicializar estado para historial de chat
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Mostrar historial previo
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Campo de texto estilo chat
    user_input = st.chat_input("Pregunta sobre el PDF...")

    if user_input:
        # Añadir mensaje del usuario al historial
        st.session_state["messages"].append({"role":"user", "content":user_input})
        st.chat_message("user").write(user_input)

        # Invocar la cadena con la pregunta
        qa_chain = st.session_state["qa_chain"]
        result = qa_chain.invoke({"query": user_input})
        answer = result["result"]

        # Añadir respuesta del asistente al historial
        st.session_state["messages"].append({"role":"assistant", "content":answer})
        st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()
