from huggingface_hub import InferenceClient
import os
from config import model_api_key_dict_name, model_path_dict

class chat_llama3_guard:
    def __init__(self):
        self.model = InferenceClient(api_key=os.getenv(model_api_key_dict_name["llama3_guard"]))
        self.system_prompt = "You are a helpful assistant."

    
    def process_input(self,role,message):
        return [{"role": role, "content": message}]
    
    def chat(self, message, max_tokens=1024):
        # Manual Configuration of Conversation History is Required
        completion = self.model.chat.completions.create(
            model="meta-llama/Llama-Guard-3-8B", 
            messages=message
        )
        return completion.choices[0].message.content
    
    def config_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
