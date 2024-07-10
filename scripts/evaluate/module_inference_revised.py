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

def load_model(model_path):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
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


def get_inference_examples(model, tokenizer, prompts, device, gen_config):
    # instruction tuning 모델 평가할 때 -> apply chat template 적용하기
    # https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template
    
    responses = []
    
    for prompt in tqdm(prompts):        
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        # add_special_tokens=False
        # apply.chat.template: add_generation_prompt=True
        outputs = model.generate(**inputs, **gen_config)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.append(response)
        
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
    parser.add_argument("--prompt_path", type=str, default="./prompt_240411.txt", help="path of the prompt to be used for inference.")
    parser.add_argument("--output_path", type=str, help="path to save the output.")
    parser.add_argument("--max_length", type=int, default=256, help="'max_length' argument for generation config.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="'max_new_tokens' argument for generation config.")
    parser.add_argument("--do_sample", default=True, action='store_false', help="'do_sample' argument for generation config.")
    parser.add_argument("--temperature", type=float, default=1.0, help="'temperature' argument for generation config.")
    parser.add_argument("--top_k", type=int, default=5, help="'top_k' argument for generation config.")
    parser.add_argument("--top_p", type=float, default=1.0, help="'top_p' argument for generation config.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="'no_repeat_ngram_size' argument for generation config.")
    parser.add_argument("--seed", type=float, default=42, help="random seed for reproducibility.")
    args = parser.parse_args()
    
    ### 1. 기본 설정
    set_seed(seed=args.seed) # 시드 고정
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    date = datetime.now().strftime("%y%m%d")
    hub = True if not os.path.isdir(args.model_path) else False
    if hub:
        model_name = os.path.split(args.model_path)[-1]        
    else:
        model_dir, checkpoint = os.path.split(args.model_path)
        model_name = os.path.split(model_dir)[-1]
        
    if not args.output_path:
        if hub:
            output_path = f"./inference_{model_name}_{date}.xlsx"
        else:
            output_path = f"./inference_{model_name}_{checkpoint}_{date}.xlsx"        
    elif args.output_path.endswith(".xlsx"):
        output_path = args.output_path
    else:
        raise ValueError("output file format should be set as '.xlsx'.")
    
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "no_repeat_ngram_size": args.no_repeat_ngram_size
    }    
    config = {
        "model path": args.model_path,
        "prompt file path": args.prompt_path,
        "output file path": output_path,
        "generation config": gen_config
    }
    print_config(config)
    
    ### 2. 모델 & 토크나이저 로드
    model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    print("len(tokenizer):", len(tokenizer))    

    ### 3. inference 시키기
    print("loading prompt info...")
    infos = get_prompt_info(args.prompt_path)
    prompts = infos[3]
    print("getting inference results...")
    responses = get_inference_examples(model, tokenizer, prompts, device, gen_config)    

    ### 4. 정성평가용 엑셀 파일 생성
    print("saving xlsx file for evaluating inference results...")
    save_inference_result(infos, config, responses, output_path)
    
    
if __name__ == "__main__":    
    main()
