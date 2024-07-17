import os
import tempfile
import streamlit as st
from streamlit_chat import message
from askmeapp_ollama import AskMe

st.set_page_config(page_title="Ask Me")

def display_message():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["message"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["message"].append((user_text, True))
        st.session_state["message"].append((agent_text, False))

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["message"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def page():
    if len(st.session_state) == 0:
        st.session_state["message"] = []
        st.session_state["assistant"] = AskMe()

    st.header("Ask Me")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )            

    st.session_state["ingestion_spinner"] = st.empty()

    display_message()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
