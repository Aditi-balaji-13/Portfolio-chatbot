# 🧠 Aditi's AI Chatbot Portfolio

This is a **RAG-based Streamlit chatbot** that answers questions about Aditi’s profile, resume, hobbies, projects, skills, and more! It’s built using:

- `Streamlit` for the frontend
- `LangChain` for orchestration
- `FAISS` as the vectorstore
- `Together AI` for the hosted LLM

You can access the app here: [Portfolio Chatbot](https://aditi-balaji-portfolio-chatbot.streamlit.app/)

Ask about:
- Aditi’s work at NASA, Goldman Sachs, or Linbeck Group
- ML books and tools she recommends
- Her AI/ML projects (GNNs, segmentation, RAG)
- Her singing, dancing, or parody songs 🎶

---

## 🔧 How It Works

1. **Documents** (`PDFs`, etc.) were embedded using `SentenceTransformer` and stored in `ChromaDB`.
2. On user query, the top relevant chunks are retrieved from Chroma.
3. These are passed to an LLM via `LangChain RetrievalQA` to generate a contextual response.

---

## 🚀 Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/<your-username>/Portfolio-chatbot.git
cd Portfolio-chatbot
```

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
