import json
import random
import uuid
import websocket
import io
from PIL import Image
from urllib import request as urllib_request, parse as urllib_parse

# 서버 주소와 클라이언트 ID 설정
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

# 프롬프트를 큐에 넣는 함수
def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib_request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(urllib_request.urlopen(req).read())

# 이미지를 가져오는 함수
def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib_parse.urlencode(data)
    with urllib_request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()

# 프롬프트 ID로 히스토리를 가져오는 함수
def get_history(prompt_id):
    with urllib_request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())

# 이미지를 가져오는 함수
def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # 실행 완료
        else:
            continue  # 미리보기는 바이너리 데이터

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

# 워크플로우 데이터를 로드하는 함수
def load_workflow():
    with open("workflow_api.json", "r", encoding="utf-8") as f:
        workflow_data = f.read()
    return json.loads(workflow_data)

# 이미지 생성 함수
def generate_image(prompt):
    try:
        # 워크플로우 로드
        workflow = load_workflow()
        # 랜덤 시드 생성
        seed = random.randint(1, 1000000000)
        # 시드를 워크플로우에 설정
        workflow["6"]["inputs"]["text"] = prompt

        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        images = get_images(ws, workflow)

        # 첫 번째 노드의 첫 번째 이미지를 반환
        for node_id in images:
            for image_data in images[node_id]:
                image = Image.open(io.BytesIO(image_data))
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)
                return buf

        raise Exception("이미지를 생성할 수 없습니다.")
    except Exception as e:
        raise Exception(str(e))
