from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("E-INVOICE-008.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} page(s)")
print("First page content preview:")
print(pages[0].page_content[:300])