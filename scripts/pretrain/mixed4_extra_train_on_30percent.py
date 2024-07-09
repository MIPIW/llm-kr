from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from trl import SFTTrainer
import pandas as pd
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import pickle

accelerator = Accelerator(gradient_accumulation_steps=256) 

torch.cuda.is_available()

path = "continual_llama_mixed4_final/checkpoint-410000"

tokenizer = AutoTokenizer.from_pretrained(path)
#tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto") 
#model.resize_token_embeddings(len(tokenizer)) 
print(path)


def preprocess_function(example):
    
    final_texts = example["Text"]

    return final_texts

def check_missing_values(example):
    for value in example.values():

        if value is None or value != value:  
            return False
    return True


dataset = load_from_disk("mixed_dataset4")
dataset = dataset.filter(check_missing_values)

print(f"{len(dataset)} lines of data ready!")



epochs = 1
train_batch_size = 16   

training_args = TrainingArguments(
    output_dir = './continual_llama_mixed4_final/',  
    save_strategy="steps",
    evaluation_strategy = "no",
    save_steps = 1000,
    learning_rate=3e-5,
    per_device_train_batch_size= train_batch_size,
    weight_decay=0.01,
    save_total_limit= 2,
    num_train_epochs= epochs,
    fp16=True,
    logging_steps = 1000,
    metric_for_best_model = 'train_loss',
    load_best_model_at_end = False,
    seed = 42,
    lr_scheduler_type = 'linear'
)


trainer = SFTTrainer(
    model,
    training_args,
    train_dataset=dataset,
    formatting_func = preprocess_function,
    max_seq_length = 2048,
    tokenizer = tokenizer
       

)

trainer.train(resume_from_checkpoint=True)
