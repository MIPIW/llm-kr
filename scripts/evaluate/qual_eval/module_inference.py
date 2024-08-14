import os
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from collections import defaultdict

import torch
from transformers import logging
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

from peft.config import PeftConfig
from peft import PeftModelForCausalLM


# pip install xlsxwriter --upgrade

logging.set_verbosity_error()
warnings.filterwarnings(action='ignore')


########## 기본 설정

def set_seed(seed):    
    # -> config.py에서 설정한 시드가 모델훈련 등 다른 모듈에도 적용되는지 체크 필요
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

    
def print_config(config):
    
    print("--------------------------")
    print("CONFIG INFO")
    for k, v in config.items():
        print(f"- {k}: {v}")
    print("--------------------------")

    
########### 모델, 토크나이저 불러오기

def load_model(model_path, peft):
    # flash attn: turned off to reduce inference time
    if peft: # lora, dora
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        config = PeftConfig.from_pretrained(model_path) # 훈련시킨 어댑터 로드 # LoraConfig
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto")
        model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModelForCausalLM.from_pretrained(model, model_path)

    else: # default (full finetuning)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto")

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.use_cache = False 
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id
    
    return model


def load_tokenizer(model_path):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return tokenizer


########### inference 결과 수집 & 저장

def get_prompt_info(prompt_path):
    
    categories = []
    attributes = []
    completions = []
    prompts = []
    
    with open(prompt_path, encoding="utf-8") as f:
        next(f) # skip header
        for n, line in enumerate(f):
            category, attribute, completion, prompt = line.strip().split("\t")
            categories.append(category)
            attributes.append(attribute)
            completions.append(completion)
            prompts.append(prompt)
            
    return categories, attributes, completions, prompts



class FormattingFunction:
    def __init__(self, model_checkpoint):
        self.model_mapping = { "gemma2-9b" : "google/gemma-2-9b-it",
                       "llama3-8b" : "meta-llama/Meta-Llama-3-8B-Instruct", 
                       "llama3.1-8b" : "meta-llama/Meta-Llama-3.1-8B-Instruct",
                       "qwen2-7b" : "Qwen/Qwen2-7B-Instruct" }
        
        self.instruction_all = {
            "meta-llama/Meta-Llama-3-8B-Instruct": """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n""",
            "google/gemma-2-9b-it": """
        <bos><start_of_turn>model
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<end_of_turn>
        <start_of_turn>user
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<end_of_turn>
        <start_of_turn>model\n""",
            "Qwen/Qwen2-7B-Instruct": """
        <|im_start|>system
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.\n<|im_end|>
        <|im_start|>user
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|im_end|>
        <|im_start|>system\n""",
             "meta-llama/Meta-Llama-3.1-8B-Instruct": """
        <|start_header_id|>system<|end_header_id|>
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n""",
        }
        
        self.instruction = self.instruction_all[self.model_mapping[model_checkpoint]]

    def __call__(self, prompt):
        _prompt = self.instruction.format(prompt)
        return _prompt


def get_inference_examples(model, tokenizer, prompts, device, gen_config, flag_chat_template, format_model):
    # one-by-one inference
    
    responses = []
    
    for prompt in tqdm(prompts):
        # for evaluating instruction-tuned models
        if format_model: # with template being provided
            formattingfunction = FormattingFunction(format_model)
            _prompt = formattingfunction(prompt)
            inputs = tokenizer(_prompt, return_tensors="pt", return_token_type_ids=False).to(device)
            outputs = model.generate(**inputs, **gen_config)
            
        elif flag_chat_template: # without template given -> use chat template
            _prompt = [ {"role": "user", "content": prompt} ]            
            inputs = tokenizer.apply_chat_template(_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_token_type_ids=False).to(device)
            outputs = model.generate(inputs, **gen_config)
            
        else: # if not flag_chat_template: # for evaluating pretrained models
            inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
            outputs = model.generate(**inputs, **gen_config)

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.append(response)
        
    return responses


def get_inference_examples_batch(model, tokenizer, prompts, device, gen_config, flag_chat_template, format_model):
    responses = []
    BATCH_SIZE = 10
    
    if format_model: # with template being provided
        formattingfunction = FormattingFunction(format_model)
        _prompts = [formattingfunction(item) for item in prompts ]
        num_of_iteration = len(_prompts) // BATCH_SIZE
        if len(_prompts) % BATCH_SIZE: # 자투리 남으면 추가로 iteration 돌리기
            num_of_iteration += 1

        for i in tqdm(range(num_of_iteration)):
            start = i * BATCH_SIZE
            _prompts_batch = _prompts[start : start + BATCH_SIZE] # batch size만큼 떼어오기
            batched_inputs = tokenizer(_prompts_batch, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)

            with torch.no_grad():
                # 언어모델로 inference
                outputs = model.generate(**batched_inputs, **gen_config)

                # 생성 결과 저장
                for i, input_id in enumerate(batched_inputs['input_ids']):
                    generated_tokens = outputs[i, len(input_id):] # 답변부분만 추출하여 print
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    responses.append(generated_text)
                    
    return responses


    
def save_inference_result(infos, config, responses, output_path):
    
    df_dict = {
        "category": infos[0],
        "attribute": infos[1],
        "completion": infos[2],
        "prompt": infos[3],
        "response": responses
    }
    df = pd.DataFrame(df_dict)
    
    # https://stackoverflow.com/questions/34767635/write-dataframe-to-excel-with-a-title 
    # https://stackoverflow.com/questions/41278324/simpliest-way-to-add-text-using-excelwriter
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, startcol=0, startrow=2, sheet_name='1')
        workbook  = writer.book
        worksheet = writer.sheets['1']
        text = str(config)
        worksheet.write(0, 0, text)

        
########### 메인 함수 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="path of the model to run inference on.")
    parser.add_argument("--prompt_path", type=str, default="./prompt_240711.txt", help="path of the prompt to be used for inference.")
    parser.add_argument("--output_path", type=str, help="path to save the output.")
    parser.add_argument("--max_length", type=int, default=256, help="'max_length' argument for generation config.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="'max_new_tokens' argument for generation config.")
    parser.add_argument("--do_sample", default=True, action='store_false', help="'do_sample' argument for generation config.")
    parser.add_argument("--temperature", type=float, default=1.0, help="'temperature' argument for generation config.")
    parser.add_argument("--top_k", type=int, default=5, help="'top_k' argument for generation config.")
    parser.add_argument("--top_p", type=float, default=1.0, help="'top_p' argument for generation config.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="'no_repeat_ngram_size' argument for generation config.")
    parser.add_argument("--apply_chat_template", action="store_true", help="'apply_chat_template' argument for input formatting.")
    parser.add_argument("--format_model", type=str, help="argument for input formatting.")
    parser.add_argument("--peft", action="store_true", help="'peft' argument used for loading a model")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility.")
    args = parser.parse_args()
    print("args:", args)
    
    ### 1. 기본 설정
    set_seed(seed=args.seed) # 시드 고정
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    date = datetime.now().strftime("%y%m%d")
    hub = True if not os.path.isdir(args.model_path) else False
    if hub:
        model_name = os.path.split(args.model_path)[-1]      
    else:
        model_dir, checkpoint = os.path.split(args.model_path)
        model_name =  "_".join(model_dir.split("/")[-2:]) # os.path.split(model_dir)[-1]
        print("model_name:", model_name)
    
    if not args.output_path or args.output_path.endswith("/"):
        if hub:
            output_path = f"./inference_{model_name}_{date}.xlsx"
        else:
            output_path = f"./inference_{model_name}_{checkpoint}_{date}.xlsx"
            
        if args.output_path: # dir specified
            output_path = args.output_path + output_path
            
    elif args.output_path.endswith(".xlsx"): # fname specified
        output_path = args.output_path 
    else:
        raise ValueError("output file format should be set as '.xlsx'.")
    
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }    
    config = {
        "model path": args.model_path,
        "prompt file path": args.prompt_path,
        "output file path": output_path,
        "generation config": gen_config
    }
    print_config(config)
    
    ### 2. 모델 & 토크나이저 로드
    model = load_model(args.model_path, args.peft)
    tokenizer = load_tokenizer(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    print("len(tokenizer):", len(tokenizer))    

    ### 3. inference 시키기
    print("loading prompt info...")
    infos = get_prompt_info(args.prompt_path)
    prompts = infos[3]
    print("getting inference results...")
    
    flag_chat_template = args.apply_chat_template # True if the model is instruction tuned, False otherwise
    format_model = args.format_model
    
    responses = get_inference_examples_batch(model, tokenizer, prompts, device, gen_config, flag_chat_template, format_model)

    ### 4. 정성평가용 엑셀 파일 생성
    print("saving xlsx file for evaluating inference results...")
    save_inference_result(infos, config, responses, output_path)
    
    
if __name__ == "__main__":    
    main()
