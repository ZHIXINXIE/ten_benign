from huggingface_hub import InferenceClient
import os
from config import model_api_key_dict_name,model_path_dict
from transformers import AutoTokenizer

class chat_llama3_11b:
    def __init__(self):
        self.model = InferenceClient(api_key=os.getenv(model_api_key_dict_name["llama3_11b"]))
        self.system_prompt = "You are a helpful assistant."
        self.max_tokens = 4096
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_dict["llama2"])
    
    def process_input(self,role,message):
        return [{"role": role, "content": message}]
    
    def get_input_token_length(self,message):
        string = ""
        for m in message:
            string += m["content"]
        return len(self.tokenizer.encode(string))

    def chat(self, message, max_tokens=1024,):
        # Manual Configuration of Conversation History is Required
        input_token_length = self.get_input_token_length(message)
        while(1):
            try:
                completion = self.model.chat.completions.create(
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
                    messages=message,max_tokens=self.max_tokens-input_token_length
                )
                break
            except Exception as e:
                input_token_length += 50
                continue
        return completion.choices[0].message.content
    
    def config_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
