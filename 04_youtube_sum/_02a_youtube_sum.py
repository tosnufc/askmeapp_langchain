from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

loop = True
while loop:
    youtube_url = input('Paste a youtube URL: ')
    if youtube_url == "/bye":
        loop = False
    else:
        loader = YoutubeLoader.from_youtube_url(f"{youtube_url}", add_video_info=False, language='en-US')

        docs = loader.load()

        transcript=docs[0].page_content
        # print(transcript)

        prompt_template = """
        You are a helpful assistant that explains Youtube Videos. Given the following video transcript:
        {video_transcript}
        Give a summary
        """

        prompt = PromptTemplate(
            input_variable=["video_transcript"],
            template=prompt_template,
        )

        # llm = ChatOllama(model='llama3.1', temperature=0)
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        chain = prompt | llm
        response = chain.invoke({'video_transcript':transcript}).content
        print(f"\n\nSummary: {response}\n\n")

        client = OpenAI()
        speech_file_path = './sum.mp3'
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova", # alloy, echo, fable, onyx, nova, shimmer
            input=response,
            response_format='mp3',
            speed=1.2,
        ) as response:
            response.stream_to_file(speech_file_path)
        print(f"sum.mp3...")
