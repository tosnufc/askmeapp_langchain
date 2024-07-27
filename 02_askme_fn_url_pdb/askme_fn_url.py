# back_end.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import ssl

load_dotenv()

ssl._create_default_https_context = ssl._create_unverified_context

def embed_url(url):
    jina_url = f"https://r.jina.ai/{url}"
    loader = WebBaseLoader(jina_url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    Chroma.from_documents(document_chunks, OpenAIEmbeddings(), persist_directory='./db/')


def retrieve():
    vector_store = Chroma(persist_directory='./db/', embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Referring to the above conversation, generate a search query relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    print(retriever_chain)
    return retriever_chain

def get_conversation(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        "system", "You are a Cisco Field Notice expert acting as an assistant. You can't answer anything that is not related to Cisco Field Notice. Answer the user's questions based on the below context:\n\n{context}. ",
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_document_chain)

def get_response(user_query):
    retriever_chain = retrieve()
    conversation_chain = get_conversation(retriever_chain=retriever_chain)
    response = conversation_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

