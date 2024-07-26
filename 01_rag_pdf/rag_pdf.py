from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# define variables for embeded
pdf_dir = './01_rag_pdf/docs'
persist_dir = './01_rag_pdf/db'
chunk_size = 1000
chunk_overlap = 150

# define additional variables for get_standalone_question
llm_model = 'gpt-4o'
temperature = 0
chain_type = 'stuff'
k = 2


def embed(data_source):
    ''' ingest pdf documents in a given folder into a vector store'''
    # load documents
    loader = PyPDFDirectoryLoader(pdf_dir)
    pre_split_documents = loader.load()

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    post_split_documents = splitter.split_documents(pre_split_documents)

    # embed the chunks into vector store
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(documents=post_split_documents,
                                embedding=embeddings,
                                persist_directory=persist_dir)

# embed(dir=pdf_dir)

def get_standalone_question():
    '''retrieve from the vector store with user's query to create a standalone question'''
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI(model=llm_model, 
                     temperature=temperature)
    retriever = vector_store.as_retriever(search_type='similarity',
                                          search_kwargs={'k':k})
    retriever_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True
    )
    return retriever_chain

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



# Example of questions (based on my documents)
# Who are Notion’s LLM Providers?
# What is FN74085?



####################
# Simple Text Chat
####################

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



####################
# Simple Tkinter GUI
####################
import tkinter as tk
from tkinter import ttk

def retrieve_input():
    input_value = text_box.get("1.0", tk.END).strip()
    get_ai_response(qa, query=input_value)
    text_box.delete("1.0", tk.END)
    output.insert(tk.END, f"YOU:  {input_value}\n\n", 'user')
    output.insert(tk.END, f"AI-BOT:  {answer}\n\n", 'bot')
    r_output.insert(tk.END, f"Question: {input_value} --> {db_query}\n", 'query')
    for doc in db_response:
        r_output.insert(tk.END, f"File: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n", 'source')
        r_output.insert(tk.END, f"{doc.page_content[:500]}.......\n", 'content')

def setup_gui_elements(root):
    global text_box, output, r_output

    text_box = tk.Text(root, height=5, width=65, font=('Helvetica', 16), wrap=tk.WORD)
    text_box.grid(row=1, column=0, padx=10, pady=10)
    text_box.bind("<Return>", on_enter_key)

    button_commit = ttk.Button(root, text="Chat", command=retrieve_input)
    button_commit.grid(row=1, column=1, padx=10, pady=10)

    scrollbar = tk.Scrollbar(root)
    output = tk.Text(root, height=20, width=65, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=('Helvetica', 16))
    output.grid(row=0, column=0, padx=10, pady=10)
    output.tag_configure('user', foreground='#3333FF')
    output.tag_configure('bot', foreground='#FF5733')
    scrollbar.grid(row=0, column=1, sticky='ns')
    scrollbar.config(command=output.yview)

    r_scrollbar = tk.Scrollbar(root)
    r_output = tk.Text(root, height=12, width=65, wrap=tk.WORD, yscrollcommand=r_scrollbar.set, font=('Helvetica', 16))
    r_output.grid(row=0, column=2, padx=10, pady=10, sticky='ns')
    r_output.tag_configure('query', foreground='#3333FF')
    r_output.tag_configure('source', foreground='#FF5733')
    r_scrollbar.grid(row=0, column=3, sticky='ns')
    r_scrollbar.config(command=r_output.yview)

def on_enter_key(event):
    retrieve_input()

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Ask Me')
    setup_gui_elements(root)
    root.mainloop()