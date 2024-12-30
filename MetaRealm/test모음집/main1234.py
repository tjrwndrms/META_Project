from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from gtts import gTTS
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
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

@app.post('/voice_stts')
async def receive_wave_file(request: Request):
    # WAV 파일 수신
    audio_bytes = await request.body()
    file_path = "received.wav"
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    
    # 음성 인식(STT) 수행
    result = pipe(audio_bytes)
    transcribed_text = result["text"]

    # STT 결과를 텍스트로 변환한 후 TTS로 처리
    text = f"{transcribed_text}"
    tts = gTTS(text=text, lang="ko")
    tts.save("output.mp3")  # gTTS는 MP3로 저장

    # MP3 파일을 PCM 형식의 WAV 파일로 변환
    sound = AudioSegment.from_mp3("output.mp3")
    pcm_wav_file_path = "output_pcm.wav"
    sound.export(pcm_wav_file_path, format="wav", parameters=["-acodec", "pcm_s16le"])

    # PCM 형식의 WAV 파일을 응답으로 반환
    return FileResponse(pcm_wav_file_path, media_type='audio/wav')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090, reload=True)