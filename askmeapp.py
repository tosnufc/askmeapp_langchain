import os
import openai
import sys
sys.path.append('../..')
import param

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma

from tkinter import *
from tkinter import ttk

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define language model name and initialize
llm_name = "gpt-3.5-turbo-0125"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
greeting = llm.invoke("Hello world!")

# Define key variables
persist_directory = 'db'
Chunk_size = 1000
Chunk_overlap = 150
K = 2
Chain_type = 'stuff'
db_flag  = 1   # 0: embed documents and save to vector database, 1: use existing vector database

def load_db(file, chain_type, k):
    loader = PyPDFDirectoryLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_size, chunk_overlap=Chunk_overlap)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True
    )
    return qa

class Cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])

    def __init__(self,  **params):
        super(Cbfs, self).__init__( **params)
        self.panels = []

    def call_load_db(self, loaded_file):
        self.loaded_file = loaded_file
        self.qa = load_db(self.loaded_file, Chain_type, K)

    def use_existing_db(self):
        embeddings2 = OpenAIEmbeddings()
        exdb = Chroma(persist_directory=persist_directory, embedding_function=embeddings2)
        exretriever = exdb.as_retriever(search_type="similarity", search_kwargs={"k": K})
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0),
            chain_type=Chain_type,
            retriever=exretriever,
            return_source_documents=True,
            return_generated_question=True,
            verbose=True)

    def convchain(self, query):
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']

cb = Cbfs()

print(db_flag)
if db_flag == 0:
    loaded_file = "docs"
    print('embed documents and save to database: '+loaded_file)
    cb.call_load_db(loaded_file)
else:
    print('use existing_db')
    cb.use_existing_db()

# Setup Tkinter GUI
root = Tk()
root.title('Ask Me')
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), background='#4CAF50', foreground='white')
style.configure('TLabel', font=('Helvetica', 12), background='white', foreground='#333')

def retrieve_input():
    inputValue = textBox.get("1.0","end-1c")
    cb.convchain(query=inputValue)
    textBox.delete("1.0", END)
    Output.insert(END, f"You: {inputValue}\n", 'user')
    Output.insert(END, f"Bot: {cb.answer}\n", 'bot')
    rOutput.insert(END, f"Question: {inputValue} --> {cb.db_query}\n", 'query')
    for doc in cb.db_response:
        rOutput.insert(END, f"File: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n", 'source')
        rOutput.insert(END, f"{doc.page_content[0:500]}.......\n", 'content')

# Setup Tkinter GUI elements
textBox = Text(root, height=5, width=65, foreground='black', background='white', font=('Helvetica', 12), wrap=WORD)
textBox.grid(row=1, column=0, padx=10, pady=10)

buttonCommit = ttk.Button(root, text="Ask", command=retrieve_input)
buttonCommit.grid(row=1, column=1, padx=10, pady=10)

scrollbar = Scrollbar(root)
Output = Text(root, height=20, width=65, foreground='black', background='white',wrap=WORD, yscrollcommand=scrollbar.set, font=('Helvetica', 12))
Output.grid(row=0, column=0, padx=10, pady=10)
Output.tag_configure('user', foreground='#3333FF')
Output.tag_configure('bot', foreground='#FF5733')
scrollbar.grid(row=0, column=1, sticky='ns')
scrollbar.config(command=Output.yview)
Output.insert(END, f"{greeting}\n", 'bot')

rscrollbar = Scrollbar(root)
rOutput = Text(root, height=12, width=65, foreground='black', background='#f0f0f0', wrap=WORD, yscrollcommand=rscrollbar.set, font=('Helvetica', 12))
rOutput.grid(row=0, column=2, padx=10, pady=10, sticky='ns')
rOutput.tag_configure('query', foreground='#3333FF')
rOutput.tag_configure('source', foreground='#FF5733')
rscrollbar.grid(row=0, column=3, sticky='ns')
rscrollbar.config(command=rOutput.yview)

# Run Tkinter event loop
root.configure(bg='white')
mainloop()
