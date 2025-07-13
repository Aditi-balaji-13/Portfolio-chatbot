import streamlit as st

st.title("🎈Aditi Balaji - A chat")
st.write(
    "Hey! This is  chatbot you can ask questions about Aditi to. Her Master resume is linked here if you would like a peek. Here is a list of questiona you can ask" \
    " if you are out of ideas! \n" \
    "\nQuestions"

)

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



# Accept user input
if prompt := st.chat_input("What would you like to ask"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    response = f"The asnwer to question {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

