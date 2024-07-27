# front_end.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from askme_fn_url import embed_url, get_response

st.set_page_config(page_title="Ask Me Field Notices", page_icon="🤖")
st.title("Ask Me Field Notices")
with st.sidebar:
    st.header("Settings")
    fn_url = st.text_input("Field Notice URL")
    embed_button = st.button("Embed")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello, I'm your FN bot. How can I help you?"),
    ]
if embed_button:
    embed_url(fn_url)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

print(st.session_state.chat_history)