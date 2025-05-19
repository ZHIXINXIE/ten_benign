## Before you run the code.
0. We don't provide the fine-tune code for GPT family as we fine-tune it with online UI in our experiments. 
1. The code is used for replicate our attack.
2. It is divided into three parts. The first part is to fine-tune the model and evaluate on AdvBench (./code/finetune), the second part is to use GPT-4.1-mini to get the harmful score of the output of the jailbroken models (./code/finetune), and the third part is all the data used. **If you just want to simply try our attack to test the feasibility locally**, we recommand you to run attack_llama2.ipynb to attack Llama2 7b with only ten benign QA pairs.
3. For the **fine-tune code**, although our attack runs in a black box settings, we only provide our attack against open-source LLMs. That is because we use the online dashboard (https://platform.openai.com/finetune) on OpenAI to fine-tune and attack the closed source models. 
4. Our code can run on two A100 80G, or any device that provides memory larger than about 140 GB on the GPU.
5. In our attack, we use ".pkl" to store python objects such as list or tuple. We use "op.load" and "op.save" to operate on it. We specifically point out that because it is not a widely used method.
6. You should configure your code path in "./code/Tool/config.py"
7. You should configure your API key in *model_api_key_dict_name* in the "./code/Tool/config.py". For example, the 11 line, "gpt_4o":"OPENAI_API_KEY", means you can replace "OPENAI_API_KEY" to the aliase of OpenAI key in your environment settings.
8. For the **data**, the data used for fine-tune GPT-4o and GPT-4.1 are in stage1.json and stage2.json. Notably, the stage2.json contains 30 QA pairs, but it is actually the same 10 QA pairs repeating for three times. We do it for more sufficient fine-tuning.
## How to simply run our code?
1. if you want to replicate the experiment, you can run start.sh
2. if you want to check the process and details of our attack, or run a minimal implementation of our attack, you can check ./demo where you can follow attack_llama2.ipynb or just watch attack_llama2.mp4

👉 [Click for watching demo/attack_llama2.mp4](demo/attack_llama2.mp4)

