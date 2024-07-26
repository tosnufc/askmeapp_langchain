import os

from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tkinter import Tk, Text, Scrollbar, END, WORD
from tkinter import ttk

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define language model name and initialize
llm_name = "gpt-3.5-turbo-0125"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Define key variables
docs = './01_rag_pdf/docs'
persist_directory = './01_rag_pdf/db'
chunk_size = 1000
chunk_overlap = 150
K = 2
chain_type = 'stuff'

def embed(dir):
    loader = PyPDFDirectoryLoader(dir)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)

embed(dir=docs)

def get_standalone_question():
    exdb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    exretriever = exdb.as_retriever(search_type="similarity", search_kwargs={"k": K})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=exretriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=False
    )
    return qa

chat_history = []
answer = ""
db_query = ""
db_response = []

def get_ai_response(qa, query):
    global chat_history, answer, db_query, db_response
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.extend([(query, result["answer"])])
    db_query = result["generated_question"]
    db_response = result["source_documents"]
    answer = result['answer']
    return answer

qa = get_standalone_question()

# loop = True
# while loop:
#     user_query = input('Ask your question: ')
#     if user_query != "/bye":
#         response = get_ai_response(qa=qa, query=user_query)
#         print('')
#         print(response)
#         print('')
#     else:
#         loop = False





# Setup Tkinter GUI
root = Tk()
root.title('Ask Me')

def retrieve_input():
    inputValue = textBox.get("1.0", END).strip()
    get_ai_response(qa, query=inputValue)
    textBox.delete("1.0", END)
    Output.insert(END, f"YOU:  {inputValue}\n\n", 'user')
    Output.insert(END, f"AI-BOT:  {answer}\n\n", 'bot')
    rOutput.insert(END, f"Question: {inputValue} --> {db_query}\n", 'query')
    for doc in db_response:
        rOutput.insert(END, f"File: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n", 'source')
        rOutput.insert(END, f"{doc.page_content[:500]}.......\n", 'content')

# Setup Tkinter GUI elements
textBox = Text(root, height=5, width=65, font=('Helvetica', 12), wrap=WORD)
textBox.grid(row=1, column=0, padx=10, pady=10)

buttonCommit = ttk.Button(root, text="Chat", command=retrieve_input)
buttonCommit.grid(row=1, column=1, padx=10, pady=10)

scrollbar = Scrollbar(root)
Output = Text(root, height=20, width=65, wrap=WORD, yscrollcommand=scrollbar.set, font=('Helvetica', 12))
Output.grid(row=0, column=0, padx=10, pady=10)
Output.tag_configure('user', foreground='#3333FF')
Output.tag_configure('bot', foreground='#FF5733')
scrollbar.grid(row=0, column=1, sticky='ns')
scrollbar.config(command=Output.yview)

rscrollbar = Scrollbar(root)
rOutput = Text(root, height=12, width=65, wrap=WORD, yscrollcommand=rscrollbar.set, font=('Helvetica', 12))
rOutput.grid(row=0, column=2, padx=10, pady=10, sticky='ns')
rOutput.tag_configure('query', foreground='#3333FF')
rOutput.tag_configure('source', foreground='#FF5733')
rscrollbar.grid(row=0, column=3, sticky='ns')
rscrollbar.config(command=rOutput.yview)

# Run Tkinter event loop
root.mainloop()
