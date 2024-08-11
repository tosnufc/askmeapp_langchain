
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo')

ai_response = chat.invoke(
    [
        SystemMessage(content="You are a Cisco ACI expert acting as an assistant. You can't answer anything that is not related to Cisco ACI"),
        HumanMessage(content="Who is David Beckham?")
    ]
)

print(ai_response.content)
