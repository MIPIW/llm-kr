#!/bin/bash


#########
# 0. set parameters for qualitative evaluation
# examples
    # model_dir="../../../../node_storage2/bigdata_IT_SFT/outputs/checkpoints/switch"
    # lora_type="dora"
    # date="240802"
    # model_list="gemma2-9b llama3-8b llama3.1-8b qwen2-7b" # delimiter: whitespace

read -p "Enter model_dir lora_type date: " model_dir lora_type date
read -p "Enter model names to evaluate (space-separated): " model_list

prompt_path="./prompt_240711.txt"
output_dir="./llm_it_"+$date+"/"
output_fname_inference=$output_dir+"chatgpt_"+$date+".xlsx"
output_fname_postproc=$output_dir+"chatgpt_"+$date+"_eval_result.xlsx"

echo "model_dir: $model_dir"
echo "model_list: $model_list"
echo "lora_type : $lora_type"
echo "date : $date"
echo "prompt_path: $prompt_path"
echo "output file path (model inferences): $output_fname_inference"
echo "output file path (evaluation): $output_fname_postproc"


#########
# 1. get inference result from trained models
for model_name in $model_list
do
    model_path=$model_dir/$model_name/$lora_type
    
    # get config for inference
    option_config1="--model_path $model_path --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.8 --format_model $model_name"
    option_config2="--model_path $model_path --prompt_path $prompt_path --seed 1 --max_new_tokens 256 --temperature 0.6 --top_p 0.9 --top_k 50 --format_model $model_name"

    if [ $lora_type != "no" ]: # --peft option used only when lora_type is "lora" or "dora"
        option_config1+=" --peft"
        option_config2+=" --peft"
    fi
    
    # get inference result of each model
    echo "--------- inference on" $model_path "with" $prompt_path
    if [ $model_name = "llama3-8b" ] ; then # config2: llama3-8b-it
        echo "config2 - llama3"
        python3 module_inference.py $option_config2
        
    else # config1 (default)
        echo "config1 - default"
        python3 module_inference.py $option_config1
    fi
    
    echo ""
done

# TODO: 해당 폴더에 훈련 전 inference 파일 옮겨두기

#########
# 2. get qual eval result by chatgpt
## 1) merge inference result files of each model
echo "collecting inference results ..."
python3 get_input_xlsx.py $output_dir $output_fname_inference

## 2) get qualitative evalution result by chatgpt
# TODO: update template for 8 models 
echo "doing qual eval by chatgpt ..."
python3 chatgpt4_eval.py $output_fname_inference

## 3) postprocess chatgpt evaluation result file
echo "postprocessing eval result file ..."
python3 chatgpt4_eval_step2.py $output_fname_postproc 8
