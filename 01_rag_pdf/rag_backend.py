from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from tkinter import Tk, Text, Scrollbar, END, WORD
from tkinter import ttk

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

loop = True
while loop:
    user_query = input('Ask your question: ')
    if user_query != "/bye":
        response = get_ai_response(qa=qa, query=user_query)
        print('')
        print(response)
        print('')
    else:
        loop = False


# Example of questions

# Who are Notion’s LLM Providers?
# What is FN74085?



# Setup Tkinter GUI
# root = Tk()
# root.title('Ask Me')

# def retrieve_input():
#     inputValue = textBox.get("1.0", END).strip()
#     get_ai_response(qa, query=inputValue)
#     textBox.delete("1.0", END)
#     Output.insert(END, f"YOU:  {inputValue}\n\n", 'user')
#     Output.insert(END, f"AI-BOT:  {answer}\n\n", 'bot')
#     rOutput.insert(END, f"Question: {inputValue} --> {db_query}\n", 'query')
#     for doc in db_response:
#         rOutput.insert(END, f"File: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n", 'source')
#         rOutput.insert(END, f"{doc.page_content[:500]}.......\n", 'content')

# # Setup Tkinter GUI elements
# textBox = Text(root, height=5, width=65, font=('Helvetica', 12), wrap=WORD)
# textBox.grid(row=1, column=0, padx=10, pady=10)

# buttonCommit = ttk.Button(root, text="Chat", command=retrieve_input)
# buttonCommit.grid(row=1, column=1, padx=10, pady=10)

# scrollbar = Scrollbar(root)
# Output = Text(root, height=20, width=65, wrap=WORD, yscrollcommand=scrollbar.set, font=('Helvetica', 12))
# Output.grid(row=0, column=0, padx=10, pady=10)
# Output.tag_configure('user', foreground='#3333FF')
# Output.tag_configure('bot', foreground='#FF5733')
# scrollbar.grid(row=0, column=1, sticky='ns')
# scrollbar.config(command=Output.yview)

# rscrollbar = Scrollbar(root)
# rOutput = Text(root, height=12, width=65, wrap=WORD, yscrollcommand=rscrollbar.set, font=('Helvetica', 12))
# rOutput.grid(row=0, column=2, padx=10, pady=10, sticky='ns')
# rOutput.tag_configure('query', foreground='#3333FF')
# rOutput.tag_configure('source', foreground='#FF5733')
# rscrollbar.grid(row=0, column=3, sticky='ns')
# rscrollbar.config(command=rOutput.yview)

# # Run Tkinter event loop
# root.mainloop()