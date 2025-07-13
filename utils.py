import os
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

GITHUB_TOKEN = os.environ.get("ITHUB_TOKEN")  # Make sure this is set in your env

REPO = "Aditi-balaji-13/chatbot_docs"
PDF_FILES = ["Master_resume.pdf", "Other.pdf"]
CHROMA_DIR = "chroma_store"

headers = {"Authorization": f"token {GITHUB_TOKEN}"}

def fetch_file_from_private_repo(repo, filepath, local_name=None):
    url = f"https://raw.githubusercontent.com/{repo}/main/{filepath}"
    if not local_name:
        local_name = os.path.basename(filepath)

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(local_name, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded {local_name}")
        return local_name
    else:
        print(f"❌ Failed to fetch {filepath}: {response.status_code}")
        return None

def build_chroma():
    all_docs = []
    for file in PDF_FILES:
        local_file = fetch_file_from_private_repo(REPO, file)
        if local_file:
            loader = PyPDFLoader(local_file)
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
