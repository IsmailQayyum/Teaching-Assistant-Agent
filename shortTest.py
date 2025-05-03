from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
print('[Tool] Extracting Document Text...')

try:
    loader = TextLoader('sa.py')
    docs = loader.load()
    if not docs:
        print("Error: No content found in the document")
    else:
        print(docs[0].page_content)
except Exception as e:
    print(f"Error loading Python file: {str(e)}")
