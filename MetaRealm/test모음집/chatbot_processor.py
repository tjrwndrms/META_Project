from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI Chatbot 설정
chat = ChatOpenAI(api_key=api_key, model_name='gpt-4o-mini')

# 대화 메모리 생성
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="The following is a conversation between a user and an AI. The AI is helpful, creative, clever, and very friendly.\n\n{chat_history}\n\nUser: {question}\nAI:",
)

llm_chain = LLMChain(llm=chat, prompt=prompt_template, memory=memory)

# 챗봇 질문 처리 함수
def process_chat(question: str) -> str:
    response = llm_chain.run(question=question)
    return response
