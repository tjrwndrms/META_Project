import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 사용할 장치 및 데이터 타입 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# STT 모델 및 프로세서 설정
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# STT 파이프라인 설정
stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# OpenAI Chatbot 설정
chat = ChatOpenAI(api_key=api_key, model_name='gpt-4o-mini')

# 프롬프트 템플릿 설정
system_prompt = PromptTemplate.from_template("""
    너는 회의에서 채팅으로 이루어진 대화를 요약해서 정리해주는 비서역할이야.
    전체 내용을 요약해서 되도록 짧게 답변하로록해
    문장을 끝낼때 '~하기로 했다.', '~하기로 한다.' 식으로 문장을 끝내줘
    Context의 내용을 잘 요약해서 한국어로 대답하도록 해.
    Context: {context}

    Answer:
""")

# LLMChain 설정
llm_chain = LLMChain(llm=chat, prompt=system_prompt)

# 바이너리 오디오 데이터를 텍스트로 변환하는 함수
def transcribe_audio(audio_bytes):
    # 바이너리 데이터를 임시 파일로 저장
    file_path = "received.wav"
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    
    # 저장된 파일을 모델에 전달하여 텍스트로 변환
    result = stt_pipeline(file_path)  # 파일 경로 전달
    return result["text"]

# 텍스트 요약 처리 함수
def summarize_text(context: str) -> str:
    response = llm_chain.run(context=context)
    return response
