import os
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma



#GITHUB_TOKEN = os.environ.get("ITHUB_TOKEN")  # Make sure this is set in your env

#REPO = "Aditi-balaji-13/chatbot_docs"
PDF_FILES = ["Maaster_resume.pdf", "Other.pdf"]
CHROMA_DIR = "chroma_store"

#headers = {"Authorization": f"token {GITHUB_TOKEN}"}



def build_chroma():
    all_docs = []
    for file in PDF_FILES:
        
        if file:
            loader = PyPDFLoader(file)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
    
    print(f"✅ Total chunks: {len(all_docs)}")

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print("✅ ChromaDB stored successfully.")

if __name__ == "__main__":
    build_chroma()
