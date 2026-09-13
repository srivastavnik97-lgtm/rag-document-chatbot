import os
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq


# Load variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY was not found. Check your .env file.")


# 1. Load PDF
loader = PyPDFLoader("E-INVOICE-008.pdf")
pages = loader.load()


# 2. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = splitter.split_documents(pages)


# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# 4. Create vector database
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="temp_collection_" + str(uuid.uuid4())
)


# 5. Ask user a question
question = input("Ask a question about the invoice: ")


# 6. Retrieve relevant chunks
results = vectorstore.similarity_search(
    question,
    k=4
)

context = "\n\n".join(
    [r.page_content for r in results]
)


# 7. Connect to Groq
client = Groq(
    api_key=GROQ_API_KEY
)


# 8. Build prompt
prompt = f"""
Answer the question using ONLY the information below.
If the answer isn't in the information, say you don't know.

Information:
{context}

Question:
{question}
"""


# 9. Send question to the LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# 10. Print answer
print(
    "\nAnswer:",
    response.choices[0].message.content
)