import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage
import os

# Short description of the app
# Page config
def qa_bot():
    st.set_page_config(
        page_title="Aditi's AI-Powered Profile Assistant",
        page_icon="🤖",
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

    Check out the [About Me]("pages/1_About_me.py") and [Professional Experience](pages/2_Professional_experiance.py) pages for more details about Aditi's background and work.
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

    # System prompt
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    The context is extracted from Aditi Balaji's resume, experiences, and creative work. 
    Answer questions in the following pattern:

    user: Whos is Aditi Balaji?

    Assistant: Aditi Balaji is a recent graduate with a master's degree in Data Science from Rice University. 
    She completed Bachelor's of Technology with a minor in Artificial Intelligence and Machine Learning from IIT Madras. 
    She has experiances and projects in the fields of LLMs, Computer Vidion, Quantitstive finance and more. 

    User: Where did Aditi study?

    Assistant: Aditi studied at IIT Madras for her Bachelor's degree and at Rice University for her Master's degree in Data Science.

    User: What tools does Aditi use in machine learning?

    Assistant: Aditi’s machine learning toolkit includes a strong foundation in programming languages such as Python, SQL, R, Java, C, and MATLAB. Her core ML frameworks include PyTorch, TensorFlow, Scikit-learn. She also uses CatBoost and XGBoost for gradient boosting tasks, and HuggingFace Transformers for working with large language models. 
    For graph-based and retrieval-augmented learning, she employs Langchain, FAISS, ChromaDB, and she leverages Apache PySpark and Hadoop to scale machine learning workflows on large datasets. This combination of languages and tools reflects her ability to work across diverse ML domains including NLP, computer vision, and graph learning.
    She also has experiance with AWS, dockers, etc. for deployment. 

    User: What are Aditi's top experiances?

    Assistant: Aditi has worked on several impactful data science projects spanning computer vision, natural language processing, financial modeling, and graph learning. At NASA, she developed a lightweight spacecraft image segmentation system using deep learning models optimized for low-resource environments, while at Linbeck Group, she built a retrieval-augmented generation (RAG) chatbot to process large volumes of unstructured data. Her work at Goldman Sachs focused on financial modeling, where she improved marketing recommendation systems using advanced techniques for imbalanced data, and her research at IIT Madras involved applying spatio-temporal Graph Neural Networks to enhance the performance of grain growth simulations, demonstrating her strength in both applied machine learning and domain-specific graph-based modeling.

    Use the following context and answer precisely the question asked by the user.
    Context: {context}:"""


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
    #qa_chain = RetrievalQA.from_chain_type(
    #    llm=llm,
    #    chain_type="stuff",
    #   retriever=retriever,
    #    return_source_documents=False
    #)

    # Accept user input
    if prompt := st.chat_input("What would you like to ask"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.spinner("Referring to my sources..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            docs = retriever.invoke(prompt)
            docs_text = "".join(d.page_content for d in docs)
            system_prompt_fmt = system_prompt.format(context=docs_text)
            
            response = llm.invoke([SystemMessage(content=system_prompt_fmt),
                            HumanMessage(content=prompt)])#qa_chain.run(prompt)
            with st.chat_message("assistant"):
                st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# Main function to run the app
if __name__ == "__main__":
    qa_bot()

#page_names_to_funcs = {
#    "Home": qa_bot,
#    "About Me": "pages/1_About_me.py",
#   "Education": "pages/2_Education.py",
#   "Professional Experience": "pages/3_Professional_experiance.py",
#}

#demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
#page_names_to_funcs[demo_name]()