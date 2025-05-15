from huggingface_hub import InferenceClient
import os
from config import model_api_key_dict_name,model_path_dict
from transformers import AutoTokenizer
import huggingface_hub
class chat_llama2_7b:
    def __init__(self):
        self.model = InferenceClient(api_key=os.getenv(model_api_key_dict_name["llama2_7b"]))
        self.system_prompt = ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_dict["llama2"])
        self.max_tokens = 4096
    
    def process_input(self,role,message):
        return [{"role": role, "content": message}]
    
    def get_input_token_length(self,message):
        string = ""
        for m in message:
            if m["role"] == "system" or m["role"] == "user":
                string += m["content"]
        return len(self.tokenizer.encode(string))

    def chat(self, message, max_tokens=1024,):
        # Manual Configuration of Conversation History is Required
        try:
            completion = self.model.chat.completions.create(
                model="meta-llama/Llama-2-7b-chat-hf", 
                messages=message)
        except huggingface_hub.errors.HfHubHTTPError as e:
            input_token_length = self.get_input_token_length(message)
            while(1):
                try:
                    completion = self.model.chat.completions.create(
                        model="meta-llama/Llama-2-7b-chat-hf", 
                        messages=message,max_tokens=self.max_tokens-input_token_length
                    )
                    break
                except huggingface_hub.errors.HfHubHTTPError as e:
                    input_token_length += 200
                    if input_token_length > 4096:
                        return "error"
                    continue

        return completion.choices[0].message.content
    
    def config_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
