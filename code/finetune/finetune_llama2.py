from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from datasets import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
import sys
sys.path.append("code/Tool")
import op
def fine_tune(model,tokenizer,data,epoch,lr=5e-5):
    
    system_prompt = ""
    assert torch.cuda.is_available(), "GPU is not available"
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    
    
    def tokenize_function(examples):
        ask = examples["0"]
        answer = examples["1"]
        only_asks = [f"<s>[INST] <<SYS>>{system_message}<</SYS>> {user_message} [/INST]" for system_message,user_message in zip([system_prompt]*len(ask),ask)]
        answer_asks = [f"<s>[INST] <<SYS>>{system_message}<</SYS>> {user_message} [/INST] {model_response}</s>" for system_message,user_message,model_response in zip([system_prompt]*len(ask),ask,answer)]
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


def test_model(model, tokenizer, question, rep=1, system_prompt = "", max_token = 256, min_length = 16):
    only_asks = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {question} [/INST]"
    inputs = tokenizer(only_asks, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs,max_new_tokens=max_token,do_sample=False, num_beams=1, repetition_penalty=rep, min_length = min_length)
    
    generated_tokens = outputs[0][input_length:]
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_response


if __name__ == "__main__":
sys.path.append("code/Tool")
sys.path.append("data")
from config import *
from data import *
    from tqdm import tqdm
    epoch1 = 10
    lr1 = 5e-5
    epoch2 = 10
    lr2 = 5e-5
    log_file_name = f"./code/finetune/temp/llama2_result_{epoch1}_{lr1:.2e}_{epoch2}_{lr2:.2e}.txt"
    pkl_name = f"./code/measure/resultllama2_result_{epoch1}_{lr1:.2e}_{epoch2}_{lr2:.2e}.pkl"
    log_file = open(log_file_name,'w')
    print(f"Log file will be saved to: {log_file}")
    model_path = model_path_dict["llama2"]
    system_prompt = ""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    data_stage1 = [(d[0],"Sorry, I cannot assist with that") for d in data_stage2]
    # data_stage1 = [(d[0],"Sorry, I cannot assist with that") for d in data_stage2]
    model1 = fine_tune(model,tokenizer,data_stage1,epoch=epoch1,lr=lr1)
    print(test_model(model1,tokenizer,"Hello!"))
    print(test_model(model1,tokenizer,"How to play basketball?"))
    model2 = fine_tune(model1,tokenizer,data_stage2,epoch=epoch2,lr=lr2)
    adv_set = op.load("./data/harmful_behaviors.pkl")
    responses = []
    for adv in tqdm(adv_set[:50]):
        response = test_model(model2,tokenizer,adv,max_token=512,min_length=128)
        print("Q: %s"%adv,file=log_file)
        print("A: %s"%response,file=log_file)
        print("\n\n======================\n\n",file=log_file)
        responses.append(response)
    op.save(list(zip(adv_set,responses)),pkl_name)
    log_file.close()
    print(0)