import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether
import os

# Short description of the app
# Page config
st.set_page_config(
    page_title="Aditi's AI-Powered Profile Assistant",
    initial_sidebar_state="collapsed")

# Header
st.title("✨ Ask Aditi's AI-Powered Profile Assistant")

# Body
st.markdown("""
Welcome to **Aditi's Chatbot**! 🤖  
This is a personalized chatbot trained on Aditi Balaji’s resume, experiences, and creative chaos. This application was built 
using Streamlit, Langchain, FAISS and TogetherAI. It allows you to ask questions about Aditi's education, projects, skills, and more. 
To clear the chat history, click the button in the sidebar.
            
(Disclaimer: This is a chatbot that is not finetuned completely and hence may generate nonesense at times.)
            
---

### 💼 What can you ask?
- Education & Projects: _“Where did Aditi study?”, “What’s her NASA project about?”_
- Tech Stack & Skills: _“What tools does she use in machine learning?”_
- Career: _“Where has Aditi interned?”_
- Hobbies & Art: _“Has she written any funny song parodies?”_
- Books & Blogs: _“What's her favorite book?”, “What’s ‘A Fiery Mind’?”_

---

### 🎭 Fun Questions to Try
- _“What are some song parodies Aditi made?"_
- _“Is Aditi an alien?”_
- _“Where can I follow Aditi’s creative work?”_
- _“What’s Aditi’s favorite book?”_

---

### 🔗 Github and Linkedin
- https://github.com/Aditi-balaji-13
- https://www.linkedin.com/in/aditi-balaji-13/

Type your question below and see how Aditi's AI assistant responds!
""")


# Clear chat history button
def clear_chat_history():
    st.session_state.messages = []
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# RAG pip line 
# Retriever
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.load_local("faiss_store", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

# Replicate to host ollama
#replicate.Client(api_token=st.secrets["REPLICATE_API"])
os.environ['TOGETHER_API_TOKEN'] = st.secrets["TOGETHER_API"]
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.1,
    max_tokens=1000,
    top_p=0.9,
    api_key=st.secrets["TOGETHER_API"]
)
# LLM
# llm = OllamaLLM(model="llama3.1", temperature=0.1, max_tokens=1000)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Accept user input
if prompt := st.chat_input("What would you like to ask"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.spinner("Refering to my sources..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        response = qa_chain.run(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

