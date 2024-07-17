#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import TextStreamer
from trl import SFTTrainer

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from accelerate import Accelerator
import argparse, logging, os
from tqdm import tqdm
import torch, random
import numpy as np

# LoRA modules
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType


# In[3]:


# dataset class
class MyDataset(Dataset):
    def __init__(self, dataset_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class FormattingFunction:
    def __init__(self, model_checkpoint):
        self.instruction_all = {
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

        self.instruction = self.instruction_all[model_checkpoint]

    def __call__(self, examples):

        final_texts = []
        for i in tqdm(range(len(examples["input"]))):
            final_text = self.instruction.format(
                examples["input"][i], examples["output"][i]
            )
            final_texts.append(final_text)

        return final_texts


# In[4]:


# model class
class ModelInitiator:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def __call__(self):
        if self.model_checkpoint == "google/gemma-2-9b-it":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        # padding 처리
        # llama3-8b
        if "Llama-3" in self.model_checkpoint:  # "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer.pad_token = tokenizer.pad_token
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            pass  # qwen2-it, gemma2-it는 pad 토큰 이미 설정되어 있음
            # qwen2: <|endoftext|> # https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/tokenizer_config.json#L36
            # gemma2: <pad>  # https://huggingface.co/google/gemma-2b-it/blob/main/tokenizer_config.json#L1511

        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer  # , self.target_modules[self.model_checkpoint]


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


# In[5]:


def main(args):
    # initialize model

    # use same tokenizer checkpoint with model
    modelInitiator = ModelInitiator(args.model)

    model, tokenizer = modelInitiator()
    seed_everything(args.seed)

    # initialize loraconfig
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config=peft_config)
    model.print_trainable_parameters()

    # prepare dataset
    # train_dataset = MyDataset(args.train_dataset_dir)
    # eval_dataset = MyDataset(args.eval_dataset_dir)

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

    # initialize training arguments
    trainingarguments = TrainingArguments(
        output_dir=args.output_dir + args.model + "/lora",
        save_strategy="steps",  # "epoch"
        eval_strategy="steps",  # "no"
        save_steps=args.save_steps,
        learning_rate=3e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        fp16=True,
        metric_for_best_model="loss",  # evaluation_strategy 설정 시 eval_loss로 평가 # "train_loss"
        load_best_model_at_end=False,
        seed=args.seed,
        lr_scheduler_type="linear",
    )

    # preparing others
    formattingfunction = FormattingFunction(
        args.model
    )  # model마다 다른 instruction template 사용하도록

    collator_fn = DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    )  # mlm=False: Autoregressive

    trainer = SFTTrainer(
        model=model,
        args=trainingarguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formattingfunction,
        max_seq_length=2048,
        tokenizer=tokenizer,
        data_collator=collator_fn,
    )

    trainer.train()


# In[6]:


types = "argumentparser"
if __name__ == "__main__":

    if types == "argumentparser":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
        parser.add_argument("--seed", default=42, type=int, required=False)
        parser.add_argument(
            "--output_dir",
            default="../../outputs/checkpoints/",
            type=str,
            required=False,
        )
        parser.add_argument("--train_batch_size", default=None, type=int, required=True)
        parser.add_argument("--eval_batch_size", default=None, type=int, required=True)
        parser.add_argument("--epochs", default=None, type=int, required=True)

        args = parser.parse_args()

    if types == "jupyter_inline":
        model_checkpoint = ""
        tokenizer_checkpoint = ""
        more_args_value = ""
        output_dir_value = ""
        train_batch_size = ""
        eval_batch_size = ""
        epochs = ""

        args = argparse.Namespace(
            model=model_checkpoint,
            tokenizer=tokenizer_checkpoint,
            output_dir=output_dir_value,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            epochs=epochs,
        )

    # custom preset
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

    if types in ["argumentparser", "jupyter_inline"]:
        main(args)


# In[ ]:




