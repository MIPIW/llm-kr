#!/bin/bash

# 1. inference 결과 파일 수합
echo "collecting inference results ..."
python3 get_input_xlsx.py llm_it_240802 llm_it_240802/chatgpt_240802.xlsx

# 2. chatgpt 평가 결과 얻기
echo "doing qual eval by chatgpt ..."
python3 chatgpt4_eval.py llm_it_240802/chatgpt_240802.xlsx

# 3. 평가 결과 파일 후처리
echo "postprocessing eval result file ..."
python3 chatgpt4_eval_step2.py llm_it_240802/chatgpt_240802_eval_result.xlsx 6
