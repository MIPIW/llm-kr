import os
import sys
import json
import random
import logging
import argparse
import warnings
import numpy as np
from time import time
from datetime import datetime
from datetime import timedelta

import torch
import transformers
from trl import SFTConfig
from trl import SFTTrainer
from peft import LoraConfig
from peft import get_peft_model
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] Initializing")


### Parsing arguments

parser = argparse.ArgumentParser(description="Instruction-Tuning")
m = parser.add_argument_group("Main Settings")
h = parser.add_argument_group("Hyperparameters")
l = parser.add_argument_group("LoRA Configs")
s = parser.add_argument_group("Save Settings")
e = parser.add_argument_group("Evaluation Settings")
o = parser.add_argument_group("Other Settings")

m.add_argument("--model", type=str, required=True, choices=["llama3-8b", "gemma2-9b", "qwen2-7b"], help="IT model type (required)")
m.add_argument("--train_dataset", type=str, default="/node_storage2/data_llm_kr/data_it_train_240724.csv", help="train dataset (should be in format of csv)")
m.add_argument("--eval_dataset", type=str, default="/node_storage2/data_llm_kr/data_it_eval_240724.csv", help="eval dataset (should be in format of csv)")
m.add_argument("--output_dir", type=str, default="./outputs/checkpoints", help="output directory for model checkpoint")

h.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
h.add_argument("--eval_batch_size", type=int, default=8, help="eval batch size")
h.add_argument("--epochs", type=int, default=1, help="# of train epochs")
h.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
h.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
h.add_argument("--lr_scheduler_type", type=str, default="linear", help="learning rate scheduler type")
h.add_argument("--warmup_ratio", type=float, default=0.0, help="warmup ratio of learning rate")
h.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for AdamW optimizer")

l.add_argument("--lora_type", type=str, default=None, choices=["lora", "dora"], help="whether to use LoRA or DoRA; defaults to None")
l.add_argument("--lora_rank", type=int, default=8, help="rank for LoRA")
l.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha for LoRA")
l.add_argument("--lora_dropout", type=float, default=0.1, help="dropout probability for LoRA")
l.add_argument("--lora_target", type=str, default="all", choices=["all", "no_qk", "no_v", "no_qkv"], help="target modules to apply LoRA; defaults to 'all'")

s.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="save strategy")
s.add_argument("--save_steps", type=int, default=500, help="save steps")
s.add_argument("--save_total_limit", type=int, default=1, help="# of last checkpoints to save")
s.add_argument("--load_best_model_at_end", action="store_true", help="whether to save best checkpoint")
s.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="metric for best checkpoint")

e.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="eval strategy")
e.add_argument("--eval_steps", type=int, default=500, help="eval steps")
e.add_argument("--eval_accumulation_steps", type=int, default=None, help="eval accumulation steps")

o.add_argument("--logging_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="logging strategy")
o.add_argument("--logging_steps", type=int, default=500, help="logging steps")
o.add_argument("--num_proc", type=int, default=8, help="# of processors to be used")
o.add_argument("--seed", type=int, default=42, help="random seed for random, numpy, torch")

args = parser.parse_args()


### Setting Logger  

output_dir = f"{args.output_dir}/{args.model}/{args.lora_type}"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)    
    
logger = logging.getLogger("main")
logger.propagate = False
logger.setLevel(logging.DEBUG)    
formatter = logging.Formatter(f"[%(asctime)s][%(levelname)s] %(message)s")

streamer = logging.StreamHandler(sys.stdout)
streamer.setFormatter(formatter)
streamer.setLevel(logging.INFO)
logger.addHandler(streamer)

filer = logging.FileHandler(filename=f"{output_dir}/train_log.log", mode="w", encoding="utf-8")
filer.setFormatter(formatter)
filer.setLevel(logging.DEBUG)   
logger.addHandler(filer)    

logger.info("Logger prepared")
logger.info(f"Logs will be documented to: {filer.baseFilename}")


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


def get_datasets(data_files):
    
    for v in data_files.values():
        if not v.endswith(".csv"):
            raise ValueError("data files should be in format of 'csv'")
    
    dataset = load_dataset("csv", sep=",", data_files=data_files, keep_default_na=False)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    
    return train_dataset, eval_dataset    
    
    
def get_trainable_parameters(model):
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    messages = [f"- all parameters: {all_params :,}",
               f"- trainable parameters: {trainable_params :,}",
               f"- trainable parameters %: {100 * trainable_params / all_params :.2f}"]
    
    return messages    


def set_lora_config():
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_target == "no_qkv":
        target_modules = target_modules[3:]
    elif args.lora_target == "no_qk":
        target_modules = target_modules[2:]
    elif args.lora_target == "no_v":
        target_modules.remove("v_proj")
    else: # args.lora_target == all
        pass
    
    lora_config_dict = {
        "use_dora": True if args.lora_type == "dora" else False,
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules
    }
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        **lora_config_dict
    )
    
    logger.debug("〓〓〓〓〓 LoRA Configuration 〓〓〓〓〓")
    for k, v in lora_config_dict.items():
        logger.debug(f"- {k}: {v}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")
    
    return lora_config


def get_model_and_tokenizer(lora_type, model_path):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"}) # Llama3 doesn't have pad_token
    
    # flash attention 2 only for Gemma
    attn_implementation = "eager" if model_path == "google/gemma-2-9b-it" else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # set LoRA config if needed    
    if lora_type:
        logger.info("Setting LoRA configuration")
        peft_config = set_lora_config()
        logger.info("Building PEFT model")
        model = get_peft_model(model, peft_config=peft_config)
        
    logger.debug("〓〓〓〓〓〓 Model Settings 〓〓〓〓〓〓")
    logger.debug(f"- model name or path: {model_path}")
    logger.debug(f"- lora type: {args.lora_type}")
    for message in get_trainable_parameters(model):
        logger.debug(message)
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")    
    
    return model, tokenizer


def set_sft_config():
    
    sft_config_dict = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay
    }
    
    # add save settings
    sft_config_dict["save_strategy"] = args.save_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'save_strategy' is set to 'no', make sure your model to be saved separately after training")
    else:
        if args.save_strategy == "steps":
            sft_config_dict["save_steps"] = args.save_steps
        sft_config_dict["save_total_limit"] = args.save_total_limit
        if args.load_best_model_at_end:
            sft_config_dict["load_best_model_at_end"] = args.load_best_model_at_end
            sft_config_dict["metric_for_best_model"] = args.metric_for_best_model
    
    # add eval settings
    sft_config_dict["eval_strategy"] = args.eval_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'eval_strategy' is set to 'no', no evaluation will be done during training")
    else:
        sft_config_dict["eval_steps"] = args.eval_steps
        sft_config_dict["eval_accumulation_steps"] = args.eval_accumulation_steps
    
    # add log settings
    sft_config_dict["logging_strategy"] = args.logging_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'logging_strategy' is set to 'no', no log will be documented")
    else:
        if args.logging_strategy == "steps":
            sft_config_dict["logging_steps"] = args.logging_steps
    
    # add other settings
    sft_config_dict["dataloader_num_workers"] = args.num_proc
    sft_config_dict["seed"] = args.seed   
    
    sft_config = SFTConfig(
        **sft_config_dict,                                      
        bf16=True,
        remove_unused_columns=True,
        group_by_length=True,
        disable_tqdm=False
    )

    logger.debug("〓〓〓〓〓 SFT  Configuration 〓〓〓〓〓")
    for k, v in sft_config_dict.items():
        logger.debug(f"- {k}: {v}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")
    
    return sft_config


def formatting_func(example):
    
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
    template = model2template[model.name_or_path].strip()
    
    formatted = []
    for i in range(len(example["input"])):
        task_input, task_output = random.choice(task_templates[example["task"][i]])
        task_input = task_input.replace("{instruction}", example['instruction'][i].strip())
        task_input = task_input.replace("{input}", example["input"][i].strip())
        task_input = task_input.replace("{option}", example["option"][i].strip())
        task_output = task_output.replace("{output}", example["output"][i].strip())
        formatted.append(template.format(task_input, task_output))
        
    return formatted        


def main():    
    
    # Setting random seed
    
    logger.info(f"Setting seed to: {args.seed}")
    set_seed(args.seed)
    

    # Loading datasets
    
    data_files = {"train": args.train_dataset, "eval": args.eval_dataset}
    for k, v in data_files.items():
        logger.info(f"Loading {k} dataset from: {v}")    
    train_dataset, eval_dataset = get_datasets(data_files)
    
    
    # Loading model & tokenizer
    
    arg2model = {
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "gemma2-9b": "google/gemma-2-9b-it",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct"
    }
    model_path = arg2model[args.model]    
    logger.info(f"Loading base model & tokenizer from: {model_path}")
    global model, tokenizer
    model, tokenizer = get_model_and_tokenizer(args.lora_type, model_path)
    
    
    # Building data collator
    
    logger.info("Building data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # mlm=False: Autoregressive


    # Loading task templates

    task_templates_path = "/node_storage2/data_llm_kr/instruction_template_240724.json"
    logger.info(f"Loading task templates from: {task_templates_path}")
    global task_templates
    with open(task_templates_path, "r", encoding="utf-8") as f:
        task_templates = json.load(f)    
        
    
    # Setting SFT config
    
    logger.info("Setting SFT configuration")
    sft_config = set_sft_config()

    
    # Building SFT trainer
    
    logger.info("Building SFT trainer")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        data_collator=data_collator,
        max_seq_length=1024,
        dataset_num_proc=args.num_proc,
    )
    
    
    # Training
    
    logger.info("Training")
    start = time()
    trainer.train()
    end = time()

    logger.debug(f"Time elapsed for training: {timedelta(seconds=round(end - start))}")    
    logger.info("Done")

    
if __name__ == "__main__":
    exit(main())
