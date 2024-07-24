import os
import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling


parser = argparse.ArgumentParser(description="Instruction-Tuning")
m = parser.add_argument_group("Main Settings")
h = parser.add_argument_group("Hyperparameters")
s = parser.add_argument_group("Save Settings")
e = parser.add_argument_group("Evaluation Settings")
o = parser.add_argument_group("Other Settings")

m.add_argument("--model", type=str, required=True, choices=["llama3-8b", "gemma2-9b", "qwen2-7b"], help="model type")
m.add_argument("--train_dataset", type=str, default="../datasets/oig-smallchip2-dedu-slice_reviewed_week1-7_instruction_train.csv", help="train dataset")
m.add_argument("--eval_dataset", type=str, default="../datasets/oig-smallchip2-dedu-slice_reviewed_week1-7_instruction_valid.csv", help="eval dataset")
m.add_argument("--output_dir", type=str, default="../outputs/checkpoints", help="output directory for model checkpoint")

h.add_argument("--train_batch_size", type=int, required=True, help="train batch size")
h.add_argument("--eval_batch_size", type=int, required=True, help="eval batch size")
h.add_argument("--epochs", type=int, required=True, help="train epochs")
h.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
h.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
h.add_argument("--lr_scheduler_type", type=str, default="linear", help="learning rate scheduler type")
h.add_argument("--warmup_ratio", type=float, default=0.0, help="warnup ratio of learning rate")
h.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for AdamW optimizer")

s.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="save strategy")
s.add_argument("--save_steps", type=int, default=500, help="save steps")
s.add_argument("--save_total_limit", type=int, default=1, help="# of last checkpoints to save")
s.add_argument("--load_best_model_at_end", action="store_true", help="whether to save best checkpoint")
s.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="metric for best checkpoint")

e.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="eval strategy")
e.add_argument("--eval_steps", type=int, default=500, help="eval steps")
e.add_argument("--eval_accumulation_steps", type=int, default=None, help="eval accumulation steps")

o.add_argument("--logging_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
o.add_argument("--logging_steps", type=int, default=500)
o.add_argument("--num_proc", type=int, default=8)
o.add_argument("--seed", type=int, default=42, help="random seed for random, numpy, torch")


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_logger(log_dir):
    
    logger = logging.getLogger("main")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)    
    formatter = logging.Formatter(f"[%(asctime)s][%(levelname)s] %(message)s")
    
    streamer = logging.StreamHandler(sys.stdout)
    streamer.setFormatter(formatter)
    streamer.setLevel(logging.INFO)
    logger.addHandler(streamer)
    
    filer = logging.FileHandler(filename=f"{log_dir}/train_log.log", mode="w", encoding="utf-8")
    filer.setFormatter(formatter)
    filer.setLevel(logging.DEBUG)   
    logger.addHandler(filer)
    
    return logger  
    

def get_model_and_tokenizer(model_path):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"}) # Llama3 doesn't have pad_token
    
    # flash attention 2 only for Gemma
    attn_implementation = "eager" if model_path == "google/gemma-2-9b-it" else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation
    )
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def formatting_func(examples):
    
    model2template = {
        "meta-llama/Meta-Llama-3-8B-Instruct": """
        <|start_header_id|>system<|end_header_id|>
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>
        <|end_of_text|>""", # post_processor adds '<|begin_of_text|>'
        "google/gemma-2-9b-it": """
        <start_of_turn>model
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<end_of_turn>
        <start_of_turn>user
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<end_of_turn>
        <start_of_turn>model
        {}<end_of_turn>
        <eos>""", # post_processor adds '<bos>'
        "Qwen/Qwen2-7B-Instruct": """
        <|im_start|>system
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.\n<|im_end|>
        <|im_start|>user
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|im_end|>
        <|im_start|>system
        {}<|im_end|>
        <|endoftext|>"""
    }
    template = model2template[model.name_or_path]
    
    formatted = []
    for i in range(len(examples["input"])):
        formatted.append(template.format(examples["input"][i], examples["output"][i]))
        
    return formatted        


def main(args):
    
    # Setting Logger
    
    output_dir = args.output_dir + "/" + args.model + "/default"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger = set_logger(output_dir)
    logger.info("Initializing")     
    
    # Setting random seed
    
    logger.info(f"Setting seed to: {args.seed}")
    set_seed(args.seed)
    

    # Loading datasets
    
    data_files = {"train": args.train_dataset, "eval": args.eval_dataset}
    for k, v in data_files.items():
        logger.info(f"Loading {k} dataset from: {v}")
    dataset = load_dataset("csv", sep=",", data_files=data_files, keep_default_na=False)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]    
    
    
    # Loading model & tokenizer
    
    arg2model = {
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "gemma2-9b": "google/gemma-2-9b-it",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct"
    }
    model_path = arg2model[args.model]
    
    global model, tokenizer
    logger.info(f"Loading model & tokenizer from: {model_path}")
    model, tokenizer = get_model_and_tokenizer(model_path)
    
    
    # Setting arguments for training
    
    logger.info("Building data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # mlm=False: Autoregressive
    
    logger.info("Setting training arguments")
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_sceduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_strategy": args.logging_strategy,
        "logging_steps": args.logging_steps,
        "dataloader_num_workers": args.num_proc,
        "seed": args.seed
    }
    
    # save settings
    training_args_dict["save_strategy"] = args.save_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'save_strategy' is set to 'no', make sure your model to be saved separately after training")
    else:
        if args.save_strategy == "steps":
            training_args_dict["save_steps"] = args.save_steps
        training_args_dict["save_total_limit"] = args.save_total_limit
        if args.load_best_model_at_end:
            training_args_dict["load_best_model_at_end"] = args.load_best_model_at_end
            training_args_dict["metric_for_best_model"] = args.metric_for_best_model
    
    # eval settings
    training_args_dict["evaluation_strategy"] = args.eval_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'eval_strategy' is set to 'no', no evaluation will be done during training")
    else:
        training_args_dict["eval_steps"] = args.eval_steps
        training_args_dict["eval_accumulation_steps"] = args.eval_accumulation_steps
    
    # log settings
    training_args_dict["logging_strategy"] = args.logging_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'logging_strategy' is set to 'no', no log will be documented")
    else:
        if args.logging_strategy == "steps":
            training_args_dict["logging_steps"] = args.logging_steps
    
    # other settings
    training_args_dict["dataloader_num_workers"] = args.num_proc
    training_args_dict["seed"] = args.seed

    logger.debug("〓〓〓〓〓 Training Arguments 〓〓〓〓〓")
    for k, v in training_args_dict.items():
        logger.debug(f"- {k}: {v}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")   
    
    training_args = TrainingArguments(
        **training_args_dict,                                      
        bf16=True,
        disable_tqdm=False
        # prediction_loss_only=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        data_collator=data_collator,
        max_seq_length=2048,
        dataset_num_proc=args.num_proc,
        # model_init_kwargs = None,
        # use_cache=False,
    )
    
    # Training
    
    logger.info("Training")
    start = time()
    trainer.train()
    end = time()

    logger.debug(f"Time elapsed for training: {timedelta(seconds=round(end - start))}")    
    logger.info("Done")

    
if __name__ == "__main__":
    exit(main(parser.parse_args()))
