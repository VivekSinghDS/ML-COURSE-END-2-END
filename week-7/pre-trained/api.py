from fastapi import FastAPI, Request
import requests
import os
from dotenv import load_dotenv 

load_dotenv()

app = FastAPI()

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")


@app.get('/')
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    prompt = body['data']['ndarray'][0]
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
    response = requests.post(HUGGING_FACE_API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return {"data": {"ndarray": [response.text]}}
    else:
        return {"error": response.text}
