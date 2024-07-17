from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import os
from transformers import DataCollatorForLanguageModeling,Trainer 
from tqdm import tqdm
import transformers
import torch
from typing import Tuple
from trl import SFTTrainer
from datasets import Dataset
import trl

from pathlib import Path
import pandas as pd
import numpy as np
import random
import json

from accelerate import Accelerator
from datasets import load_dataset

from transformers import TrainingArguments
from transformers import AutoTokenizer 
from datasets import load_metric



def get_model_and_tokenizer(args) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    model_name = args.model_name


    # Load tokenizer
    accelerator = Accelerator(gradient_accumulation_steps=256)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token = tokenizer.pad_token
    tokenizer.add_special_tokens({"pad_token":"<pad>"})

    # Load huggingface model
    hf_model =AutoModelForCausalLM.from_pretrained(args.model_name,
    device_map="auto",trust_remote_code=True,   attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,)
    hf_model.resize_token_embeddings(len(tokenizer))   
    
    return hf_model, tokenizer

def train(args):

    # Loading config, model and tokenizer
    hf_model, tokenizer = get_model_and_tokenizer(args)

    # Setting random seed
    seed_everything(args.random_seed)


    dataset = load_dataset('csv', delimiter = ',', data_files = {'train' : [args.instruction_datasets], 'validation' : [args.instruction_datasets]}, keep_default_na=False)
    train_dataset = dataset['train']
    # Loading Dataset
    ## Train Dataset
    
    epochs = 1
    train_batch_size =1
    eval_batch_size =1
    #SAVE_STEP = 10600
    training_args = TrainingArguments(
        output_dir = './update/',
        save_strategy="epoch",
        evaluation_strategy = "no",
        #save_steps = SAVE_STEP,
        learning_rate=3e-5,
        per_device_train_batch_size= train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit= 1,
        num_train_epochs= epochs,
    #    fp16=False,
        bf16=True,
        logging_steps = 1,
        metric_for_best_model = 'train_loss',
        load_best_model_at_end = False,
        seed = 42,
        lr_scheduler_type = 'linear'
    #    prediction_loss_only=True,
    )
    def preprocess_function(example):
        
        final_texts = []
        for i in tqdm(range(len(example['input']))):
                    
            final_text= f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

    아래 질문을 한국어로 정확하게 답변해주세요.
    **질문**: {example['input'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['output'][i]}<|eot_id|><|end_of_text|>"""
    #        print(final_text)
            final_texts.append(final_text)
        return final_texts


    trainer = SFTTrainer(
            hf_model,
        training_args,
        train_dataset=train_dataset,
        formatting_func = preprocess_function,
        max_seq_length = 2048,
        tokenizer=tokenizer,
        #model_init_kwargs = None
    #    use_cache=False
    )

    trainer.train()


   
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Instruction-Tuning")

    # Dataset names
    parser.add_argument("--instruction_datasets", type=str, default="", help="instruction datasets")
    parser.add_argument("--dataset_sizes", type=str, default="[1.0,10%]", help="instruction dataset ratios")
    parser.add_argument("--evaluation_datasets", type=str, default="", help="")

    # Random Seed
    parser.add_argument("--random_seed", type=int, default=42, help="fix random seed in torch, random, numpy")

    # Data & Logging Path
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="model checkpoint path")

    # Model & Tokenizer path
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="path for evaluation prediction results")

    args = parser.parse_args()

    args.run_name = f"MODEL_NAME:{args.model_name}-DATASETS:{args.instruction_datasets}"
    args.output_dir = f"{args.output_dir}"

    train(args)
