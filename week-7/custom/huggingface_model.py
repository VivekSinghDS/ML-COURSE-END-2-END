from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
#https://api-inference.huggingface.co/models/microsoft/phi-2
class HuggingFaceModel:
    def __init__(self):
        model_name = 'microsoft/phi-2'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, use_auth_token=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left",
            add_eos_token=True, add_bos_token=True, use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, X, feature_names):
        def gen(model, p, maxlen=100, sample=True):
            toks = self.tokenizer(p, return_tensors="pt")
            res = model.generate(
                **toks.to("cpu"), max_new_tokens=maxlen, 
                do_sample=sample, num_return_sequences=1, 
                temperature=0.1, num_beams=1, top_p=0.95
            ).to('cpu')
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)

        prompt = X[0]
        result = gen(self.model, prompt, 5)
        return result