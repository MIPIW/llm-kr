from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
import os
from transformers import DataCollatorForLanguageModeling, Trainer
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

    # model = args.model

    # Load tokenizer
    # accelerator = Accelerator(gradient_accumulation_steps=256)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if "Llama-3" in args.model:
        tokenizer.pad_token = tokenizer.pad_token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Load huggingface model
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    hf_model.resize_token_embeddings(len(tokenizer))

    return hf_model, tokenizer


def train(args):

    # Loading config, model and tokenizer
    hf_model, tokenizer = get_model_and_tokenizer(args)

    # Setting random seed
    seed_everything(args.seed)

    dataset = load_dataset(
        "csv",
        delimiter=",",
        data_files={
            "train": [args.train_dataset_dir],
            "validation": [args.eval_dataset_dir],
        },
        keep_default_na=False,
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    # Loading Dataset
    ## Train Dataset

    # SAVE_STEP = 10600
    training_args = TrainingArguments(
        output_dir=args.output_dir + args.model + "/default",
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=3e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        bf16=True,
        metric_for_best_model="loss",
        load_best_model_at_end=False,
        seed=args.seed,
        lr_scheduler_type="linear",
        #    prediction_loss_only=True,
    )

    def preprocess_function(example, model_checkpoint):
        ###### different instruction template per models
        instruction_all = {
            "meta-llama/Meta-Llama-3-8B-Instruct": """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>
        <|end_of_text|>""",
            "google/gemma-2-9b-it": """
        <bos><start_of_turn>model
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<end_of_turn>
        <start_of_turn>user
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<end_of_turn>
        <start_of_turn>model
        {}<end_of_turn>
        <eos>""",
            "Qwen/Qwen2-7B-Instruct": """
        <|im_start|>system
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.\n<|im_end|>
        <|im_start|>user
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|im_end|>
        <|im_start|>system
        {}<|im_end|>
        <|endoftext|>""",
        }

        final_texts = []
        for i in tqdm(range(len(example["input"]))):
            final_text = instruction_all[model_checkpoint].format(
                example["input"][i], example["output"][i]
            )
            final_texts.append(final_text)
        return final_texts

    collator_fn = DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    )  # mlm=False: Autoregressive

    trainer = SFTTrainer(
        hf_model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda x: preprocess_function(x, args.model),
        max_seq_length=2048,
        tokenizer=tokenizer,
        # model_init_kwargs = None
        #    use_cache=False
        data_collator=collator_fn,
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

    # Model & Tokenizer path
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="path for evaluation prediction results",
        required=True,
    )

    # Random Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="fix random seed in torch, random, numpy",
    )

    # Data & Logging Path
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../outputs/checkpoints/",
        help="model checkpoint path",
    )

    parser.add_argument("--train_batch_size", default=None, type=int, required=True)
    parser.add_argument("--eval_batch_size", default=None, type=int, required=True)
    parser.add_argument("--epochs", default=None, type=int, required=True)

    args = parser.parse_args()

    train_dataset_dir = (
        "../../data/oig-smallchip2-dedu-slice_reviewed_week1-7_instruction_train.csv"
    )
    eval_dataset_dir = (
        "../../data/oig-smallchip2-dedu-slice_reviewed_week1-7_instruction_valid.csv"
    )
    args.train_dataset_dir = train_dataset_dir
    args.eval_dataset_dir = eval_dataset_dir
    args.save_steps = 500
    args.logging_steps = 500

    print(args.logging_steps)
    print(args.seed)

    # 어떤 용도?
    # args.run_name = f"model:{args.model}-DATASETS:{args.eval_dataset_dir}"
    args.output_dir = f"{args.output_dir}"

    train(args)
