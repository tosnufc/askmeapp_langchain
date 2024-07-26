from tkinter import Tk, Text, Scrollbar, END, WORD
from tkinter import ttk
from rag_backend import get_ai_response, qa, answer, db_query, db_response

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