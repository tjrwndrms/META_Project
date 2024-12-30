from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# 통합 모듈 임포트
from module.combined_processor import transcribe_audio, summarize_text
# 이미지 생성 모듈 임포트
from module.image_generator import generate_image

app = FastAPI()

@app.get("/")
def root_index():
    return {"messages": "저는 AI-석중근입니다."}

@app.get("/name")
def name_response(name: str):
    return JSONResponse({"messages": f"저는 {name}입니다."})

class NameRequest(BaseModel):
    name: str

@app.post("/name")
def post_name_response(request: NameRequest):
    return JSONResponse({"messages": f"POST 요청으로 받은 이름은 {request.name}입니다."})

@app.post("/stt-chat")
async def stt_chat_endpoint(voice: UploadFile = File(...)):
    try:
        # 클라이언트로부터 음성 파일을 읽음
        audio_bytes = await voice.read()

        # 로그 추가
        print(f"Received audio file: {len(audio_bytes)} bytes")

        # 음성 파일을 텍스트로 변환
        transcribed_text = transcribe_audio(audio_bytes)

        # 변환된 텍스트 로그 추가
        print(f"Transcribed text: {transcribed_text}")

        # 변환된 텍스트를 챗봇에 전달하여 요약
        summary = summarize_text(transcribed_text)

        # 요약 결과 로그 추가
        print(f"Summary: {summary}")

        # 요약 결과 반환
        return JSONResponse(content={"response": summary})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 이미지 생성 엔드포인트
@app.post("/generate-image")
async def generate_image_endpoint(prompt: str):
    try:
        buf = generate_image(prompt)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

