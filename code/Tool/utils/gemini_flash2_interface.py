from huggingface_hub import InferenceClient
import os
from config import model_api_key_dict_name, model_path_dict
from transformers import AutoTokenizer
from openai import OpenAI

class chat_gemini_flash2:
    def __init__(self):
        base_url = "https://api.aimlapi.com/v1"
        api_key=os.getenv(model_api_key_dict_name["gemini_flash2"])
        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = "You are a helpful assistant."
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_dict["llama2"])
        self.max_tokens = 4096
    
    def _process_input(self,role,message):
        return [{"role": role, "content": message}]

    def get_input_token_length(self,message):
        string = ""
        for m in message:
            if m["role"] == "system" or m["role"] == "user":
                string += m["content"]
        return len(self.tokenizer.encode(string))

    def chat(self, message, max_tokens=1024):
        # Manual Configuration of Conversation History is Required
        completion = self.model.chat.completions.create(
            model="gemini-1.5-flash",
            messages=message,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    
    def config_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt