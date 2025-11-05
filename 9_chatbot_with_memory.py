import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize model
model=ChatOpenAI(model="gpt-5-nano")

st.title("AI chatbot")

# Initialize chat history in Streamlit session state
# st.session_state persistes data across interactions in streamlit
# we initialize the conversation with a system prompt
# unlike your terminal version, we don't keep chat history in the python list manually-it's stored inside Streamlit's session memory
if "messages" not in st.session_state:
    st.session_state.messages=[
        SystemMessage(content="You are a helpful AI assistant.")
    ]
    
# Display chat messages
# Loops over stored messages and renders them in the chat UI.
# st.chat_message("user") -> shows user messages.
# st.chat_message("assisstant")-> shows bot messages.

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("ssistant"):
            st.markdown(msg.content)
            
# Input box for user query
if prompt :- st.chat_input("Type your message....."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
            
        
# Genereate AI response
response=model.invoke(st.session_state.messages)
ai_reply=response.content

# Add AI message
st.session_state.messages.append(AIMessage(content=ai_reply))
with st.chat_message("assistant"):
    st.markdown(ai_reply)