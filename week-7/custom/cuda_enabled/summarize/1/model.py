import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import triton_python_backend_utils as pb_utils
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import json 
import numpy as np
class TritonPythonModel:
    def initialize(self, args):
        """
        Load the model and tokenizer during initialization.
        """
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        base_model_id = "microsoft/phi-2"
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map='auto',
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # Load fine-tuned model
        self.model = PeftModel.from_pretrained(base_model, 
            "/models/summarize/1/checkpoint-350",
            torch_dtype=torch.float16,
            is_trainable=False
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, 
            trust_remote_code=True,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def gen(self, prompt, maxlen=100, sample=True):
        """
        Generate text using the model.
        """
        toks = self.tokenizer(prompt, return_tensors="pt")
        res = self.model.generate(
            **toks.to("gpu"), 
            max_new_tokens=maxlen, 
            do_sample=sample, 
            num_return_sequences=1, 
            temperature=0.1, 
            num_beams=1, 
            top_p=0.95
        ).to('gpu')
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)
    
    def execute(self, requests):
        """
        This function will be called by Triton Inference Server.
        """
        responses = []
        for request in requests:
            # Get the input tensors (assuming a string input here)
            input_data = pb_utils.get_input_tensor_by_name(request, "INPUT_DATA").as_numpy().tolist()
            
            # Deserialize the input data if necessary
            input_dict = json.loads(input_data[0])
            dialogue = input_dict.get("dialogue")
            # Prepare the input for model generation
            prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"
            
            try:
                # Call your fine-tuned model
                peft_model_res = self.gen(self.model, prompt, 5)
                peft_model_output = peft_model_res[0].split('Output:\n')[1]
                prefix, success, result = peft_model_output.partition('###')
            except Exception as e:
                result = f"Model generation failed: {str(e)}"
            
            # Create an output tensor
            output_data = json.dumps({"data": {"ndarray": [peft_model_res]}})
            output_tensor = pb_utils.Tensor("OUTPUT_DATA", np.array([output_data], dtype=object))

            # Create and send the inference response
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses