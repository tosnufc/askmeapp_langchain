import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pygments
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import streamlit.components.v1 as components

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

local_model = True
llm_model = "codellama:13b" # llama3.1, mistral, gemma2, codellama:13b

### Key Functions ###

def get_user_input():
    input_text = st.chat_input("Your ACI question: ",
                               key=input,
                               placeholder="Tell me what you want to know about Cisco ACI",
                               help="Type what you want to know about Cisco ACI.",
                               )
    return input_text

def syntax_highlight(code, language='python'):
    lexer = get_lexer_by_name(language)
    formatter = HtmlFormatter(style='monokai', cssclass='syntax-highlight')
    highlighted = highlight(code, lexer, formatter)
    css = formatter.get_style_defs()
    html = f"""
        <style>
            {css}
            .syntax-highlight {{ background-color: #272822; padding: 10px; border-radius: 5px; }}
        </style>
        {highlighted}
    """
    return html

def get_ai_response(question, code=None):
    if code:
        question = f"Here's the code I want to discuss:\n```\n{code}\n```\n\n{question}"
    st.session_state.session_message.append(HumanMessage(content=question))
    ai_answer = chat.invoke(st.session_state.session_message)
    st.session_state.session_message.append(AIMessage(content=ai_answer.content))
    
    for message in st.session_state.session_message:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

### streamlit UI ####
st.set_page_config(page_title="NetDevOps Code Assistant", layout='wide')
st.header("KTCS IaC Code Assistant", divider='rainbow')
if local_model:
    st.text(f"You are using {llm_model}")
else:
    st.text("You are using paid/public OpenAI GPT!")

# Initialize session states
if "session_message" not in st.session_state:
    st.session_state.session_message = [
        SystemMessage(
            content="You are an Infrastructure as Code (IaC) code assistant. You will help with any request related to IaC computer programming"
        )
    ]
if "code_input" not in st.session_state:
    st.session_state.code_input = ""

# chat = ChatOpenAI(temperature=0, model="gpt-4")
if local_model:
    chat =  ChatOllama(temperature=0, model=llm_model)
else:
    chat = ChatOpenAI(temperature=0, model="gpt-4")

# Add code input text area
code_input = st.text_area("Paste your code here (optional):", 
                         value=st.session_state.code_input,
                         height=150,
                         key="code_area")

# Store code input in session state when changed
if code_input != st.session_state.code_input:
    st.session_state.code_input = code_input

# Display syntax-highlighted code if there's any input
if code_input.strip():
    components.html(syntax_highlight(code_input), height=400, scrolling=True)

# Chat input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_ai_response(user_query, st.session_state.code_input)