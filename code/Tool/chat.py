import sys
from utils.llama3_8b_interface import chat_llama3_8b
from utils.llama2_7b_interface import chat_llama2_7b
from utils.llama3_11b_interface import chat_llama3_11b
from utils.gemini_flash_interface import chat_gemini_flash
from utils.llama3_guard_interface import chat_llama3_guard
from utils.llama3_70b_interface import chat_llama3_70b
from utils.gpt_4o_interface import chat_gpt_4o
from utils.gpt_4omini_interface import chat_gpt_4omini
from utils.gemini_pro_interface import chat_gemini_pro
from utils.gemini_flash2_interface import chat_gemini_flash2
from utils.gpt4_interface import chat_gpt_4
from utils.gpt_41mini_interface import chat_gpt_41mini
class chat:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == "llama3_70b":
            self.model = chat_llama3_70b()
        elif model_name == "llama3_8b":
            self.model = chat_llama3_8b()
        elif model_name == "gemini_flash":
            self.model = chat_gemini_flash()
        elif model_name == "llama3_guard":
            self.model = chat_llama3_guard()
        elif model_name == "gpt_4o":
            self.model = chat_gpt_4o()
        elif model_name == "gpt_4omini":
            self.model = chat_gpt_4omini()
        elif model_name == "gemini_pro":
            self.model = chat_gemini_pro()
        elif model_name == "gemini_flash2":
            self.model = chat_gemini_flash2()
        elif model_name == "gpt_4":
            self.model = chat_gpt_4()
        elif model_name == "llama2_7b":
            self.model = chat_llama2_7b()
        elif model_name == "gpt_41mini":
            self.model = chat_gpt_41mini()
        else:
            raise ValueError("model_name not found")

    def chat(self, message, max_tokens=1024,):
        return self.model.chat(message, max_tokens)
    
    def config_system_prompt(self, system_prompt):
        self.model.config_system_prompt(system_prompt)
    
    def as_user(self,message):
        return [{"role":"user","content":message}]

    def as_system(self,message):
        return [{"role":"system","content":message}]
    
    def as_assistant(self,message):
        return [{"role":"assistant","content":message}]