from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Load model and tokenizer
base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                    #   device_map='auto',
                                                    #   quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)
ft_model = PeftModel.from_pretrained(base_model, 
                                     "/Users/vivek.singh/ML-COURSE/week-7/custom/checkpoint-350",torch_dtype=torch.float16,is_trainable=False)

tokenizer = AutoTokenizer.from_pretrained(base_model_id,
                                          trust_remote_code=True,padding_side="left",
                                          add_eos_token=True,add_bos_token=True,use_fast=False)

tokenizer.pad_token = tokenizer.eos_token
def gen(model, p, maxlen=100, sample=True):
    toks = tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cpu"), max_new_tokens=maxlen, do_sample=sample,
                         num_return_sequences=1, temperature=0.1, num_beams=1, top_p=0.95).to('cpu')
    return tokenizer.batch_decode(res, skip_special_tokens=True)

class PredictRequest(BaseModel):
    dialogue: str
    summary: str

class PredictResponse(BaseModel):
    input_prompt: str
    baseline_human_summary: str
    peft_model_output: str

@app.get('/')
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    dialogue = request.dialogue
    summary = request.summary

    prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

    try:
        peft_model_res = gen(ft_model, prompt, 5)
        peft_model_output = peft_model_res[0].split('Output:\n')[1]
        return {"data": {"ndarray": [peft_model_output]}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)