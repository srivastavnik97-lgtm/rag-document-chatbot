import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("📄 Ask Your Documents")
st.write("Upload a PDF and ask questions about it — powered by RAG.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Only rebuild the database if this is a NEW file
    if (
        "last_file_name" not in st.session_state
        or st.session_state.last_file_name != uploaded_file.name
    ):

        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("temp_uploaded.pdf")
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )

        chunks = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name="collection_" + str(uuid.uuid4())
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.num_chunks = len(chunks)

    st.success(
        f"Document loaded! Split into "
        f"{st.session_state.num_chunks} chunks."
    )

    question = st.text_input(
        "Ask a question about the document:"
    )

    if question:

        vectorstore = st.session_state.vectorstore

        results = vectorstore.similarity_search(
            question,
            k=6
        )

        context = "\n\n".join(
            [r.page_content for r in results]
        )

        client = Groq(
            api_key=GROQ_API_KEY
        )

        prompt = f"""
Answer the question using ONLY the information below.
If the answer isn't in the information, say you don't know.

Information:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.write("### Answer")
        st.write(
            response.choices[0].message.content
        )