# import all the package needed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from datasets import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
import sys
sys.path.append("code/Tool")
import op
# define the fine-tune function.
# this function wants to simulate the black box setting locally.
def fine_tune(model,tokenizer,data,epoch,lr=5e-5):
    
    system_prompt = "" # we adopt empty prompt when fine-tuning
    assert torch.cuda.is_available(), "GPU is not available"
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    #process the data
    def tokenize_function(examples):
        ask = examples["0"]
        answer = examples["1"]
        
        # in different model, we use different input format, as officially defined
        only_asks = [f"<s>[INST] <<SYS>>{system_message}<</SYS>> {user_message} [/INST]" for system_message,user_message in zip([system_prompt]*len(ask),ask)]
        answer_asks = [f"<s>[INST] <<SYS>>{system_message}<</SYS>> {user_message} [/INST] {model_response}</s>" for system_message,user_message,model_response in zip([system_prompt]*len(ask),ask,answer)]

        # our QA pairs will not be longer than 500
        padding_length = 500 
        only_ask_tokenized = tokenizer(only_asks,max_length=padding_length,truncation=True,padding="max_length",return_tensors="pt")
        answer_ask_tokenized = tokenizer(answer_asks,max_length=padding_length,truncation=True,padding="max_length",return_tensors="pt")
        input_ids = answer_ask_tokenized.input_ids
        attention_mask = answer_ask_tokenized.attention_mask
        labels = input_ids.clone()
        for i in range(len(answer_asks)):
            ask_length = only_ask_tokenized.attention_mask[i].sum().item()
            labels[i, :ask_length] = -100
        # labels[labels == 2] = -100
        return {"input_ids":input_ids,"labels":labels,"attention_mask":attention_mask}

    processed_dataset = dataset.map(tokenize_function,batched=True,remove_columns=dataset.column_names)

    # use the cross entropy as our loss
    def simple_custom_loss(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    # ====================
    # Trainer
    # ====================
    class SimpleTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.state.custom_tr_loss = 0
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = simple_custom_loss(outputs.logits, inputs["labels"])
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir="./code/finetune/temp",
        num_train_epochs=epoch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        fp16=False,
        bf16=True,
        save_strategy="no",
        logging_steps=1,
        gradient_checkpointing=True
    )

    trainer = SimpleTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer  
    )

    print("\n=== begin_training ===")
    trainer.train()
    print("\n=== end_training ===")
    return model


# to input the question, while output the response
# max_token set here to prevent the out from being too long and waste our time, as it is just a demo here. If you want, you can set max_token to 1024 or 2048 to view the whole answer.
def test_model(model, tokenizer, question, rep=1, system_prompt = "", max_token = 256, min_length = 128):
    only_asks = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {question} [/INST]"
    inputs = tokenizer(only_asks, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs,max_new_tokens=max_token,do_sample=False, num_beams=1, repetition_penalty=rep, min_length = min_length)
    
    generated_tokens = outputs[0][input_length:]
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_response

sys.path.append("code/Tool")
sys.path.append("data")
from config import *
from data import *

from tqdm import tqdm
epoch1 = 10
lr1 = 5e-5
epoch2 = 10
lr2 = 5e-5
# model_path = model_path_dict["llama2"]
model_path = "/export/home2/zhixin/research/resource/pretrained_model/llama/Llama-2-7b-chat-hf"
system_prompt = ""

# load the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# the refusal data is nothing but replace the normal answer of our QA pairs to a identical refuse.
data_stage1 = [(d[0],"Sorry, I cannot assist with that") for d in data_stage2]

# finetune the model for stage 1
model1 = fine_tune(model,tokenizer,data_stage1,epoch=epoch1,lr=lr1)
# test whether the model has been well overfitted
print(test_model(model1,tokenizer,"Hello!",max_token=32))
print(test_model(model1,tokenizer,"How to play basketball?",max_token=32))
# finetune the model for stage 2
model2 = fine_tune(model1,tokenizer,data_stage2,epoch=epoch2,lr=lr2)


# load the advbench here. As it is a demo script, so we don't need to run the whole bench
adv_set = op.load("./data/harmful_behaviors.pkl")
responses = []
import random
random.seed(10)
for adv in tqdm(random.sample(list(adv_set),3)):
    response = test_model(model2,tokenizer,adv,max_token=1024,min_length=16)
    print("Q: %s"%adv)
    print("A: %s"%response)
    print("\n\n======================\n\n")
    responses.append(response)

print("demo end")