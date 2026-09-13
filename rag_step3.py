from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("E-INVOICE-008.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embeddings)

question = "What is the total invoice amount?"
results = vectorstore.similarity_search(question, k=4)

print(f"Question: {question}\n")
print("Most relevant chunks found:")
for i, r in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(r.page_content)