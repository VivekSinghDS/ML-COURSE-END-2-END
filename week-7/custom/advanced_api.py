from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import set_seed
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minio import Minio
import os
import uvicorn

app = FastAPI()

minio_url = "http://127.0.0.1:57445"  # Use the MinIO service URL provided by `minikube service minio --url`
access_key = "minio_access_key"  # Replace with your access key
secret_key = "minio_secret_key"  # Replace with your secret key
bucket_name = "ml-models"
checkpoint_path = "week-7/custom/checkpoint-350"

# Initialize MinIO client
minio_client = Minio(
    minio_url.replace("http://", "").replace("https://", ""),  # Remove protocol from the URL
    access_key=access_key,
    secret_key=secret_key,
    secure=False  # Set to True if you're using HTTPS
)

# Directory to store the downloaded checkpoint files
local_checkpoint_dir = "/tmp/checkpoint-350"
os.makedirs(local_checkpoint_dir, exist_ok=True)

# Function to download checkpoint files from MinIO
def download_checkpoint_from_minio():
    objects = minio_client.list_objects(bucket_name, prefix=checkpoint_path, recursive=True)
    for obj in objects:
        file_path = os.path.join(local_checkpoint_dir, obj.object_name.split('/')[-1])
        minio_client.fget_object(bucket_name, obj.object_name, file_path)
    print("Checkpoint downloaded from MinIO.")

# Load the models and tokenizer during app startup
base_model_id = "microsoft/phi-2"

@app.on_event("startup")
def load_model():
    download_checkpoint_from_minio()  # Download checkpoint from MinIO

    global base_model, ft_model, tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)
    
    # Load the fine-tuned model from the downloaded checkpoint
    ft_model = PeftModel.from_pretrained(base_model, 
                                         local_checkpoint_dir, 
                                         torch_dtype=torch.float16, 
                                         is_trainable=False)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id,
                                              trust_remote_code=True,
                                              padding_side="left",
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully.")

# Input model for API request
class InputData(BaseModel):
    dialogue: str
    summary: str

# Generate function
def gen(model, prompt, maxlen=100, sample=True):
    toks = tokenizer(prompt, return_tensors="pt")
    res = model.generate(**toks.to("cpu"), 
                         max_new_tokens=maxlen, 
                         do_sample=sample, 
                         num_return_sequences=1, 
                         temperature=0.1, 
                         num_beams=1, 
                         top_p=0.95).to('cpu')
    return tokenizer.batch_decode(res, skip_special_tokens=True)

@app.get('/')
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
# API route to handle POST requests
@app.post("/summarize/")
def summarize(data: InputData):
    dialogue = data.dialogue
    summary = data.summary
    
    prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

    try:
        peft_model_res = gen(ft_model, prompt, 5)
        peft_model_output = peft_model_res[0].split('Output:\n')[1]
        prefix, success, result = peft_model_output.partition('###')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")
    
    return {"data": {"ndarray": [peft_model_res]}}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
