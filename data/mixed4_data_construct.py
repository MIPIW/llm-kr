from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from trl import SFTTrainer
import pandas as pd
from accelerate import Accelerator
from datasets import load_dataset
import glob
import pickle
import random
from datasets import concatenate_datasets, Dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from trl import SFTTrainer
from datasets import load_dataset
import random
from datasets import Dataset, interleave_datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
import csv
login("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")


tokenizer = AutoTokenizer.from_pretrained("llama2_bymistral_240403")
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset
ko_files = [
    "./kin_med.csv"
]


ko_dataset = load_dataset('csv', data_files={'train': ko_files}, split='train')

# 결측치 확인 및 제거를 위한 함수 #TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]] 방지
def check_missing_values(example):
    for value in example.values():
        if value is None or value != value:  
            return False
    return True


ko_dataset = ko_dataset.filter(check_missing_values)

en_files = [
    "./Multi_news_train_2.csv",
]

en_ko_dataset = load_dataset('csv', data_files={'train': en_files}, split='train')
en_ko_dataset = en_ko_dataset.filter(check_missing_values)

def preprocess_function(example):
    text = example["Text"] if "Text" in example else "Missing Text"
    return tokenizer(text, truncation=True, padding='max_length', max_length=2048)


ko_processed_dataset = ko_dataset.map(preprocess_function, batched=True)
en_ko_processed_dataset = en_ko_dataset.map(preprocess_function, batched=True)

ko_processed_dataset = ko_processed_dataset.shuffle(seed=42)
en_ko_processed_dataset = en_ko_processed_dataset.shuffle(seed=42)


mixed_dataset = interleave_datasets([ko_processed_dataset, en_ko_processed_dataset], probabilities=[0.95, 0.05], seed=42)
mixed_dataset.save_to_disk("/node_storage1/mixed_dataset4")
