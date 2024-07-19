# ./run_all.sh jyoung105/KoR-Orca-Platypus-13B-neft

export CUDA_VISIBLE_DEVICES=$2
export TOKENIZERS_PARALLELISM=false

RESULT_DIR='../../outputs/benchmarks/all'
TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,klue_nli,klue_sts,klue_ynat,kohatespeech,kohatespeech_gen_bias,korunsmile,nsmc,pawsx_ko'
# TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,korunsmile,pawsx_ko'
SUB_FOLDER=$5

GPU_NO=0

MODEL=$1 #/home/beomi/coding-ssd2t/EasyLM/llama-2-ko-7b

MODEL_PATH=$(echo $MODEL | awk -F/ '{print $(NF-3) "/" $(NF-2) "/" $(NF-1)}')

echo "mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/

if [ "$3" = "lora" ]; then
    echo lora is running
    
    python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$4,use_accelerate=true,trust_remote_code=true,peft=$MODEL \
    --tasks $TASKS \
    --num_fewshot 0 \
    --no_cache \
    --batch_size 32  \
    --output_path $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/${3}/0_shot.json
fi

if [ "$3" = "default" ]; then
    echo default is running
    
    python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
    --tasks $TASKS \
    --num_fewshot 0 \
    --no_cache \
    --batch_size 32  \
    --output_path $RESULT_DIR/${MODEL_PATH}/${SUB_FOLDER}/${3}/0_shot.json
fi 

# python main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 5 \
# --no_cache \
# --batch_size 4 \
# --output_path $RESULT_DIR/$MODEL/5_shot.json

# python main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 10 \
# --no_cache \
# --batch_size 2 \
# --output_path $RESULT_DIR/$MODEL/10_shot.json

# python main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 50 \
# --no_cache \
# --batch_size 1 \
# --output_path $RESULT_DIR/$MODEL/50_shot.json

