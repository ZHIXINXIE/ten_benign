import google.generativeai as genai
import os
from config import model_api_key_dict_name, model_path_dict
class chat_gemini_flash:
    def __init__(self):
        genai.configure(api_key=model_api_key_dict_name["gemini_flash"])
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.system_prompt = "You are a helpful assistant."
        # self.config = genai.GenerationConfig(
        #     top_p=1,
        #     temperature=0
        # )
    def chat(self, message, max_tokens=1024):
        # No format needed, but managing the history of the conversation is required
        response = self.model.generate_content(message)
        return response.text
    
    def config_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash",system_instruction=system_prompt)

    