from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models  # 추가된 부분
from PIL import Image
import io
import base64

# FastAPI 앱 초기화
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "백엔드 이미지 인식 전송"}

# CNN 모델 클래스 정의
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(-1, 64 * 28 * 28)          # Flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2  # 예시: 클래스 수
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()  # 평가 모드로 설정

# 클래스 이름 리스트
class_names = ["can", "pet"]  # 클래스 순서에 맞게 이름을 지정하세요

# 이미지 전처리 함수 정의
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    이미지를 딥러닝 모델에 입력할 수 있도록 전처리하는 함수.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가

    return image

# JSON 데이터 모델 정의
class ImageData(BaseModel):
    image_data: str

# 예측을 수행하는 엔드포인트
@app.post("/predict")
async def predict_image(image_data: ImageData):
    """
    클라이언트로부터 이미지를 JSON으로 받아서 딥러닝 모델로 예측값을 반환하는 엔드포인트.
    """
    try:
        # base64로 인코딩된 이미지를 디코딩하여 PIL 이미지로 변환
        image_bytes = base64.b64decode(image_data.image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 이미지 전처리
        processed_image = preprocess_image(image).to(device)
        
        # 모델 예측 수행
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probabilities, 1)
        
        # 임계값 0.5 적용
        if max_prob.item() < 0.5:
            prediction = "no trash image"
        else:
            # 예측된 클래스 이름 가져오기
            prediction = class_names[preds.item()]
        
        # 예측 결과를 JSON 형태로 반환
        return JSONResponse(content={"prediction": prediction})
    
    except Exception as e:
        # 예외 발생 시 에러 메시지 반환
        return JSONResponse(content={"error": str(e)}, status_code=500)

# class FineTunedResNet18(nn.Module):
#     def __init__(self, num_classes):
#         super(FineTunedResNet18, self).__init__()
#         # ResNet-18 모델 로드
#         self.resnet18 = models.resnet18(weights='DEFAULT')
#         num_ftrs = self.resnet18.fc.in_features
#         self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         return self.resnet18(x)

# # 모델 로드
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = 2  # 예시: 클래스 수
# model = FineTunedResNet18(num_classes=num_classes)
# model.load_state_dict(torch.load("fine_tuned_resnet18_additional.pth", map_location=device), strict=False)
# model.to(device)
# model.eval()

# # 클래스 이름 리스트
# class_names = ["can", "pet-bottle"]

# # 이미지 전처리 함수 정의
# def preprocess_image(image: Image.Image) -> torch.Tensor:
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = transform(image)
#     image = image.unsqueeze(0)  # 배치 차원 추가
#     return image

# # JSON 데이터 모델 정의
# class ImageData(BaseModel):
#     image_data: str

# # 예측을 수행하는 엔드포인트
# @app.post("/predict")
# async def predict_image(image_data: ImageData):
#     try:
#         # base64로 인코딩된 이미지를 디코딩하기 전에 헤더 제거
#         if image_data.image_data.startswith('data:image'):
#             image_base64 = image_data.image_data.split(',')[1]
#         else:
#             image_base64 = image_data.image_data

#         # 디코딩
#         image_bytes = base64.b64decode(image_base64)
#         print("Image bytes decoded successfully")

#         # 디코딩된 바이트 데이터를 파일로 저장
#         with open("decoded_image.png", "wb") as f:
#             f.write(image_bytes)
#         print("Image saved as decoded_image.png")

#         # PIL 이미지로 변환 시도 및 RGB로 변환
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         print("Image opened and converted to RGB successfully")

#         # 이미지 전처리
#         processed_image = preprocess_image(image).to(device)
#         print("Image preprocessed successfully")
        
#         # 모델 예측 수행
#         with torch.no_grad():
#             outputs = model(processed_image)
#             probabilities = F.softmax(outputs, dim=1)  # 확률로 변환
#             max_prob, preds = torch.max(probabilities, 1)
#         print("Model prediction made successfully")

#         # 예측된 클래스 이름 가져오기
#         predicted_class = class_names[preds.item()]
#         print(f"Prediction: {predicted_class} with probability {max_prob.item()}")

#         # 임계값 설정 (예: 0.6)
#         threshold = 0.5

#         # 예측이 임계값을 넘지 않는 경우 "no trash image"로 처리
#         if max_prob.item() < threshold:
#             return JSONResponse(content={"prediction": "no trash image"})
#         else:
#             return JSONResponse(content={"prediction": predicted_class})
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# OpenAI Chatbot 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(api_key=api_key, model_name='gpt-4o-mini')

# 단계별로 안내할 텍스트를 관리하기 위한 메모리
class StepMemory:
    def __init__(self):
        self.step = 0
        self.steps = [
            "페트병 분리수거를 도와줄게! 첫 번째로, 페트병의 내용물을 완전히 비워줘야 해.",
            "좋아, 이제 두 번째로, 페트병을 압축해서 부피를 줄여보자!",
            "세 번째로, 가능한 경우 라벨을 제거해줘. 하지만 일부 지역에서는 그대로 둬도 괜찮아.",
            "이제 마지막으로, 뚜껑을 분리해서 페트병과 따로 배출해야 해. 그리고 지정된 재활용 통에 넣으면 돼!"
        ]

    def get_next_step(self):
        if self.step < len(self.steps):
            response = self.steps[self.step]
            self.step += 1
            return response
        else:
            # 모든 단계를 완료하면 자동 초기화
            self.reset()
            return "페트병 분리수거에 대한 모든 단계를 완료했어! 초기화되어 다시 처음부터 시작할 수 있어."
    
    def reset(self):
        self.step = 0

# 단계를 관리하는 메모리 인스턴스 생성
step_memory = StepMemory()

# 대화 메모리 생성
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template="""
    당신은 분리수거 전문가입니다. 아래는 이전 대화 내용입니다:

    {chat_history}

    사용자 질문에 대해 최선의 분리수거 가이드를 제공하세요.

    질문: {question}

    답변:
    """
)

llm_chain = LLMChain(llm=chat, prompt=prompt_template, memory=memory)

# 분리수거 가이드를 제공하는 새로운 엔드포인트
class GuideRequest(BaseModel):
    question: str

@app.post("/recycle-guide")
async def recycle_guide(request: GuideRequest):
    try:
        # 단계별로 가이드를 제공
        response_text = step_memory.get_next_step()

        # JSON 형태로 응답 반환
        return JSONResponse(content={"guidance": response_text})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 단계 리셋 엔드포인트 (원하는 경우 단계 초기화)
@app.post("/reset-steps")
async def reset_steps():
    try:
        step_memory.reset()
        return JSONResponse(content={"message": "단계를 초기화했습니다."})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

