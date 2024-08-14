# pip install -U transformers
import sys
import os
import random
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import time
from tqdm import tqdm

from ast import literal_eval

# 소요시간: 샘플 260개 & batch size 32-> 약 3분 (182.30 sec)
# eta: 약 10만 건 & batch size 8-> 100시간 (batch size 32: OOM)
# 커맨드 예시: python reformulate_output.py data_it_240724_output_reformulate_1.csv 8 816  # row.Index 기준 816번 행부터 이어서 돌릴 때.


# seed 설정 함수
def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

# 모델, 토크나이저 로드 함수
def load_model(model_path):

    fp_dict = {"google/gemma-2-9b-it": torch.float16, "Qwen/Qwen2-7B-Instruct": torch.bfloat16}

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=fp_dict[model_path],
            attn_implementation="flash_attention_2"
    )
    
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    return model


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    
    return tokenizer


# 모델별 프롬프트 포맷팅 클래스
class FormattingFunction:
    def __init__(self, model_checkpoint):
        self.instruction_all = {
            "google/gemma-2-9b-it": """
        <bos><start_of_turn>model
        당신의 역할은 한국어로 답변하는 한국어 AI 어시트턴트입니다. 주어진 질문에 대해 한국어로 답변해주세요.<end_of_turn>
        <start_of_turn>user
       아래 문장을 읽기 편하게 개행 문자와 볼드체를 활용하여 구조화된 형태로 바꿔주세요.\n\n{}<end_of_turn>
        <start_of_turn>model\n""",
            "Qwen/Qwen2-7B-Instruct": """
        <|im_start|>system
        당신의 역할은 한국어로 답변하는 한국어 AI 어시트턴트입니다. 주어진 질문에 대해 한국어로 답변해주세요.\n<|im_end|>
        <|im_start|>user
        아래 문장을 읽기 편하게 개행 문자, 글머리 기호, 볼드체를 활용하여 구조화된 형태로 바꿔주세요. 이때 말투는 공손한 존칭을 사용해주세요.\n\n{}<|im_end|>
        <|im_start|>system\n""",
        }

        self.instruction = self.instruction_all[model_checkpoint]

    def __call__(self, prompt):
        _prompt = self.instruction.format(prompt)
        return _prompt



if __name__ == "__main__":
    set_seed(seed=42) # 시드 고정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    
    if len(sys.argv) != 4:
        sys.exit(f"python {sys.argv[0]} fname_input batch_size start_idx")
        
    F_INPUT_PATH = sys.argv[1] # F_INPUT_PATH = "data_it_240724_output_reformulate.csv" # "inference_sample_240731.csv"
    BATCH_SIZE = int(sys.argv[2]) # 16
    START_IDX = int(sys.argv[3])
    ############
    # constant 설정
    MODEL_PATH = "Qwen/Qwen2-7B-Instruct"
    F_OUTPUT_PATH = F_INPUT_PATH[:-4] + "_output.tsv"
    MAX_LENGTH = 4096 # oom -> input truncation 추가
    ############
    
    # 1. 모델 로드
    print("loading model & tokenizer...")
    model_path = MODEL_PATH
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    model.resize_token_embeddings(len(tokenizer))
    
    if model_path == "Qwen/Qwen2-7B-Instruct":
        model.config.pad_token_id = tokenizer.pad_token_id

    # 로드한 모델에 따른 포맷팅 객체 설정
    formattingfunction = FormattingFunction(model_path)

    # 2. 생성옵션 설정
    gen_config = {
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 5,
        "no_repeat_ngram_size": 0,
    }  

    # 3. 파일 i/o 준비
    # input (재구조화 전)
    df = pd.read_csv(F_INPUT_PATH)
    tasks = df["task"]
    output_raw = df["output"]

    # output (재구조화 전후 비교)
    f_o = open(F_OUTPUT_PATH, "w", encoding="utf-8") 
    header = "index\toutput\toutput_rvsd"
    print(header, file=f_o)

    # 4. output 재구조화 (batch inference)
    _prompts = [formattingfunction(item) for item in output_raw ]
    num_of_iteration = (len(_prompts) // BATCH_SIZE)
    if len(_prompts) % BATCH_SIZE:
        num_of_iteration += 1
    
    start_iter = START_IDX // BATCH_SIZE
    start_time = time.time()
    
    for i in tqdm(range(num_of_iteration)):
        if i < start_iter:
            continue
            
        start = i * BATCH_SIZE
        _prompts_batch = _prompts[start : start + BATCH_SIZE] # batch size만큼 떼어오기
        
        # truncation 적용: https://huggingface.co/docs/transformers/en/pad_truncation
        # batched_inputs = tokenizer(_prompts_batch, padding=True, return_tensors="pt", return_token_type_ids=False, ).to(device)
        # truncate & batch의 max length로 패딩
    
        batched_inputs = tokenizer(_prompts_batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", return_token_type_ids=False).to(device)
        
        
        with torch.no_grad():
            # 언어모델로 inference
            outputs = model.generate(**batched_inputs, **gen_config)
    
            # 생성 결과 출력
            generated_texts = []
            for i, input_id in enumerate(batched_inputs['input_ids']):
                generated_tokens = outputs[i, len(input_id):] # 답변부분만 추출하여 print
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            for idx, gen_text in enumerate(generated_texts):
                _idx = start + idx
                _task = tasks[_idx]
                input_text = output_raw[_idx] # 재구조화 전
                print(_idx, _task, repr(input_text), repr(gen_text), sep="\t", file=f_o)
                
    print(f"=========================== total generation took {time.time()-start_time:.2f} sec. ===========================")
    f_o.close()






