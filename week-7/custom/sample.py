from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login

interpreter_login()
compute_dtype = getattr(torch, "float16")
# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type='nf4',
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=False,
#     )

model_name='microsoft/phi-2'
# device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    #   device_map=device_map,
                                                    #   quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",
                                          add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def gen(model,p, maxlen=100, sample=True):
    toks = tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cpu"), max_new_tokens=maxlen, 
                         do_sample=sample,num_return_sequences=1,temperature=0.1,num_beams=1,top_p=0.95,).to('cpu')
    return tokenizer.batch_decode(res,skip_special_tokens=True)

prompt = "what is 1 + 1"
res = gen(original_model, prompt, 5)
print(res)
# prompt = "The moon, Earth's only natural satellite, has captivated humans for millennia with its ethereal glow and changing phases. Orbiting our planet at an average distance of about 238,900 miles, it exerts a powerful influence on Earth's tides and has played a crucial role in the evolution of life on our planet. The moon's surface is a rugged landscape of craters, mountains, and vast plains called maria, formed by ancient volcanic activity and countless meteor impacts. In 1969, it became the first celestial body beyond Earth to be visited by humans when Apollo 11 astronauts Neil Armstrong and Buzz Aldrin made their historic landing. Despite being our closest cosmic neighbor, the moon still holds many mysteries, and continues to be a subject of scientific exploration and a source of inspiration for artists, poets, and dreamers alike."
# summary = "About moon"

# formatted_prompt = f"Instruct: Summarize the following conversation.\n{prompt}\nOutput:\n"
# res = gen(original_model,formatted_prompt,100,)
# #print(res[0])
# output = res[0].split('Output:\n')[1]

# dash_line = '-'.join('' for x in range(100))
# print(dash_line)
# print(f'INPUT PROMPT:\n{formatted_prompt}')
# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
# print(dash_line)
# print(f'MODEL GENERATION - ZERO SHOT:\n{output}')