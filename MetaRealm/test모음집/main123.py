from fastapi import FastAPI
from fastapi.responses import FileResponse
from api_calltest import load_image


app = FastAPI()

@app.get('/')
def root_index():
    return 'Hello World'

@app.post('/image/{input}')
def gen_img(input:str):
    img_path = load_image(input)
    return FileResponse(img_path)