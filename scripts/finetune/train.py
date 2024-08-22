import os
import sys
import json
import logging
import argparse
import warnings
import math
import random
import numpy as np
from glob import glob
from time import time
from datetime import datetime
from datetime import timedelta
from shutil import rmtree

import wandb
import torch
from trl import SFTConfig
from trl import SFTTrainer
from peft import LoraConfig
from peft import get_peft_model
from datasets import load_dataset
##########################################ADDED###############################################
from transformers.trainer import *

"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
# isort: off
from .integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint
from .models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
    is_torch_greater_or_equal_than_2_3,
)
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_ipex_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
    strtobool,
)
from .utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        is_mlu_available,
        is_mps_available,
        is_npu_available,
        is_torch_version,
        is_xpu_available,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

##########################################ADDED###############################################

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

from lr_scheduler_utils import SplitLRSchedulerLoader

##########################################ADDED###############################################
from peft import AdaLoraModel, AdaLoraConfig
##########################################ADDED###############################################


warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

timestamp = datetime.now()
print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] Initializing")

### Parsing arguments

parser = argparse.ArgumentParser(description="Instruction-Tuning")
m = parser.add_argument_group("Main Settings")
h = parser.add_argument_group("Hyperparameters")
p = parser.add_argument_group("PEFT Configs")
l = p = parser.add_argument_group("Optimizer and Learning Rate Scheduler Configs")
s = parser.add_argument_group("Save Settings")
e = parser.add_argument_group("Evaluation Settings")
o = parser.add_argument_group("Other Settings")

m.add_argument("--model", type=str, required=True, choices=["llama3-8b", "llama3.1-8b", "gemma2-9b", "qwen2-7b"], help="IT model type (required)")
m.add_argument("--train_dataset", type=str, nargs="+", default=["/node_storage2/data_llm_kr/data_it_rpd_train_240806.csv", "/node_storage2/data_llm_kr/data_it_qa_train_240806.csv"], help="train datasets, could be more than 1 (should be in format of csv)")
m.add_argument("--eval_dataset", type=str, nargs="+", default=["/node_storage2/data_llm_kr/data_it_rpd_eval_240806.csv", "/node_storage2/data_llm_kr/data_it_qa_eval_240806.csv"], help="eval datasets, could be more than 1 (should be in format of csv)")
m.add_argument("--task_templates", type=str, default="/node_storage2/data_llm_kr/instruction_template_240806.json", help="task template")
m.add_argument("--output_dir", type=str, default="./outputs/checkpoints", help="output directory for model checkpoint")

h.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
h.add_argument("--eval_batch_size", type=int, default=8, help="eval batch size")
h.add_argument("--epochs", type=int, default=1, help="# of train epochs")
h.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
h.add_argument("--max_sequence_length", type=int, default=2048, help="max sequence length")

p.add_argument("--peft_type", type=str, default="no", choices=["no", "lora", "dora"], help="whether to use LoRA or DoRA; defaults to 'no'")
p.add_argument("--lora_rank", type=int, default=8, help="rank for LoRA")
p.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha for LoRA")
p.add_argument("--lora_dropout", type=float, default=0.1, help="dropout probability for LoRA")
p.add_argument("--lora_target", type=str, default="all", choices=["all", "no_qk", "no_v", "no_qkv"], help="target modules to apply LoRA; defaults to 'all'")

l.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
l.add_argument("--lr_scheduler_type", type=str, default="linear", help="learning rate scheduler type")
l.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of learning rate warmup steps to total steps")
l.add_argument("--warmup_steps", type=int, default=0, help="learning rate warmup steps, overwrites warmup ratio if set")
l.add_argument("--min_lr_ratio", type=float, default=0.0, help="ratio of minimum learning rate to original learning rate")
l.add_argument("--min_lr", type=float, default=0.0, help="minimum learning rate, overwrites minimun learning rate ratio if set")
l.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for AdamW optimizer")
l.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon(ε) for AdamW optimizer")

s.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="save strategy")
s.add_argument("--save_steps", type=int, default=500, help="save steps")
s.add_argument("--save_total_limit", type=int, default=1, help="# of last checkpoints to save")
s.add_argument("--save_per_dataset", action="store_true", help="whether to save all checkpoints from train datasets")
s.add_argument("--load_best_model_at_end", action="store_true", help="whether to save best checkpoint")
s.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="metric for best checkpoint")

e.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="eval strategy")
e.add_argument("--eval_steps", type=int, default=500, help="eval steps")
e.add_argument("--eval_accumulation_steps", type=int, default=None, help="eval accumulation steps")

o.add_argument("--logging_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="logging strategy")
o.add_argument("--logging_steps", type=int, default=500, help="logging steps")
o.add_argument("--report_to_wandb", action="store_true", help="whether to report logs to wandb, overwrite logging settings if set")
o.add_argument("--num_proc", type=int, default=8, help="# of processors to be used")
o.add_argument("--seed", type=int, default=42, help="random seed for random, numpy, torch")

args = parser.parse_args()


### Setting Logger  

output_dir = f"{args.output_dir}/{args.model}/{args.peft_type}"
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

logger.info("Logger is prepared")
logger.info(f"Logs will be documented to: {filer.baseFilename}")


def set_seed(seed):

    logger.info(f"Setting seed to: {args.seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_datasets(train_dataset, eval_dataset):

    # Check if the # of train/eval dataset(s) are valid
    if len(train_dataset) == 1:
        if len(eval_dataset) > 1: # 1 train, 2≤ eval
            raise ValueError("2 or more eval datasets are given while 1 train dataset is given.")
    elif len(eval_dataset) == 1: # 2≤ train, 1 eval
        logger.warning("1 eval dataset is given while 2 or more train datasets are given, same eval dataset will be applied for all phases of training.")
    elif len(train_dataset) != len(eval_dataset): # 2≤ train, 2≤ eval, train ≠ eval
        raise ValueError(f"{len(eval_datasets)} eval datasets are given while {len(train_dataset)} train datasets are given.")
    else: # 2≤ train, 2≤ eval, train = eval
        logger.warning("2 or more train and eval dataset are given, make sure if train and eval datasets are aligned.")

    train_data_files = {}
    for i, dataset in enumerate(train_dataset):
        if not dataset.endswith(".csv"):
            raise ValueError("All datasets should be in format of 'csv'")
        logger.info(f"Loading train dataset #{i+1} from {dataset}")
        train_data_files[f"train{i+1}"] = dataset
    train_dataset_dict = load_dataset("csv", sep=",", data_files=train_data_files, keep_default_na=False)
    
    eval_data_files = {}
    for i, dataset in enumerate(eval_dataset):
        if not dataset.endswith(".csv"):
            raise ValueError("All datasets should be in format of 'csv'")
        logger.info(f"Loading eval dataset #{i+1} from {dataset}")
        eval_data_files[f"eval{i+1}"] = dataset
    eval_dataset_dict = load_dataset("csv", sep=",", data_files=eval_data_files, keep_default_na=False)

    train_dataset_list = list(train_dataset_dict.values())
    eval_dataset_list = list(eval_dataset_dict.values())
    
    return train_dataset_list, eval_dataset_list    
    
    
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
        "use_dora": True if args.peft_type == "dora" else False,
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules
    }
    

    ##########################################ADDED###############################################
    adalora_config_dict = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        # in huggingface, this line is descripted as "target_modules = [ 'q', 'k', 'v' ]", not with casual lora setting: "target_modules = [ 'q_proj', 'k_proj', 'v_proj' ]"
        # 그치만 일단 그냥 넣었습니다
        "target_modules": target_modules 
        
    }
    ##########################################ADDED###############################################

    lora_config = LoraConfig(task_type="CAUSAL_LM", **lora_config_dict)
    ##########################################ADDED###############################################
    adalora_config = AdaLoraConfig(peft_type = "ADALORA", task_type="CAUSAL_LM", **adalora_config_dict)
    ##########################################ADDED###############################################


    logger.debug("〓〓〓〓〓 LoRA Configuration 〓〓〓〓〓")
    for k, v in lora_config_dict.items():
        logger.debug(f"- {k}: {v}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")

    return lora_config, adalora_config
    


def get_tokenizer(model_path):

    logger.info(f"Loading tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"}) # Llama3 doesn't have pad_token

    return tokenizer


def get_model(peft_type, model_path, load_count):

    if load_count < 2 :
        logger.info(f"Loading base model from: {model_path}")
    else:
        logger.info(f"Loading checkpoint from: {model_path}")
    
    # no flash attention 2 only for Gemma
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
    if not peft_type == "no":
        logger.info("Setting LoRA configuration")
        peft_config, adalora_config = set_lora_config()
        logger.info("Building PEFT model")
        model = get_peft_model(model, peft_config=peft_config) # lora
        ##########################################ADDED###############################################
        # model = AdaLoraModel(model, adalora_config, "default") # if lora_type == "adalora"
        ##########################################ADDED###############################################

    if load_count < 2:
        logger.debug("〓〓〓〓〓〓 Base Model Info. 〓〓〓〓〓〓")
        logger.debug(f"- model name or path: {model_path}")
        logger.debug(f"- peft type: {peft_type}")
        for message in get_trainable_parameters(model):
            logger.debug(message)
        logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")    
    
    return model


def set_sft_config():

    logger.info("Setting SFT configuration")
    
    sft_config_dict = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    
    # add save settings
    sft_config_dict["save_strategy"] = args.save_strategy
    if args.save_strategy == "no":
        logger.warning("Argument 'save_strategy' is set to 'no', make sure your model to be saved separately after training.")
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
        logger.warning("'eval_strategy' is set to 'no', no evaluation will be done during training.")
    else:
        sft_config_dict["eval_steps"] = args.eval_steps
        sft_config_dict["eval_accumulation_steps"] = args.eval_accumulation_steps

    # add wandb setting
    if args.report_to_wandb:
        logger.warning("'report_to_wandb' is set to True, 'logging_strategy' and 'logging_steps' will be overwritten with 'steps' and 1.")
        sft_config_dict["logging_strategy"] = "steps"
        sft_config_dict["logging_steps"] = 1
        sft_config_dict["report_to"] = "wandb"
        data_switch = "switch-" if len(args.train_dataset) > 1 else ""
        sft_config_dict["run_name"] = f"{args.peft_type}-{args.lr_scheduler_type}-{data_switch}it"
    else:
        # add log settings    
        sft_config_dict["logging_strategy"] = args.logging_strategy
        if args.logging_strategy == "no":
            logger.warning("'logging_strategy' is set to 'no', no log will be documented during training.")
        elif args.logging_strategy == "steps":
            sft_config_dict["logging_steps"] = args.logging_steps
        else: # args.logging_strategy == "epoch"
            pass
    
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


def get_lr_scheduler_loader(steps_list):

    total_steps = sum(steps_list)
    min_lr_ratio = args.min_lr / args.learning_rate if args.min_lr > 0.0 else args.min_lr_ratio
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else math.ceil(args.warmup_ratio * total_steps)

    logger.info("Building lr_scheduler loader")
    lr_scheduler_loader = SplitLRSchedulerLoader(
        steps_list=steps_list,
        learning_rate=args.learning_rate,
        optimizer_type="adamw",
        lr_scheduler_type=args.lr_scheduler_type,
        min_lr_ratio=min_lr_ratio,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon
    )

    logger.debug("〓〓〓 LRScheduler Configuration 〓〓〓")
    logger.debug(f"- learning rate: {lr_scheduler_loader.learning_rate}")
    logger.debug(f"- optimizer type: {lr_scheduler_loader.optimizer_type}")
    logger.debug(f"- scheduler type: {lr_scheduler_loader.lr_scheduler_type}")
    logger.debug(f"- # of lr_schedulers: {lr_scheduler_loader.max_count}")
    logger.debug(f"- # of steps in total: {lr_scheduler_loader.total_steps}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")

    return lr_scheduler_loader


def formatting_func(example):
    
    model2template = {
        "meta-llama/Meta-Llama-3-8B-Instruct": """
        <|start_header_id|>system<|end_header_id|>
        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>
        <|end_of_text|>""", # post_processor adds '<|begin_of_text|>'
        "meta-llama/Meta-Llama-3.1-8B-Instruct": f"""
        <|start_header_id|>system<|end_header_id|>

        Cutting Knowledge Date: December 2023
        Today Date: {timestamp.strftime('%d %b %Y')}

        당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.<|eot_id|>
        """
        + """<|start_header_id|>user<|end_header_id|>
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



##########################################Copied from huggingface trainer.py###############################################
class MyTrainer(SFTTrainer):
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.optimizer.step()

                    ##########################################ADDED###############################################
                    model.update_and_allocate(self.state.global_step)
                    ##########################################ADDED###############################################

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
##########################################Copied from huggingface trainer.py###############################################

def main():

    # Setting random seed

    logger.info(f"Setting seed to: {args.seed}")

    set_seed(args.seed)


    ### Wandb login if needed

    if args.report_to_wandb:
        wandb.login()
        os.environ["WANDB_PROJECT"] = f"bigdata-{args.model}-it"
    

    ### Loading datasets
    
    train_dataset_list, eval_dataset_list = get_datasets(args.train_dataset, args.eval_dataset)
    
    
    ### Loading model & tokenizer
    
    arg2model = {
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gemma2-9b": "google/gemma-2-9b-it",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct"
    }
    model_path = arg2model[args.model]    
    global tokenizer
    tokenizer = get_tokenizer(model_path)
    
    
    # Building data collator
    
    logger.info("Building data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    # Loading task templates

    logger.info(f"Loading task templates from: {args.task_templates}")
    global task_templates
    with open(args.task_templates, "r", encoding="utf-8") as f:
        task_templates = json.load(f)    
        
    
    # Setting SFT config
    
    sft_config = set_sft_config()


    # Building lr_scheduler loader

    batch_size = (sft_config.per_device_train_batch_size
                  * sft_config.gradient_accumulation_steps
                  * sft_config.num_train_epochs)    
    steps_list = [math.ceil(len(dataset) / batch_size) for dataset in train_dataset_list]
    lr_scheduler_loader = get_lr_scheduler_loader(steps_list)
    

    # Training
    
    logger.info("Training")    
    total_start = time()    
    for i in range(lr_scheduler_loader.max_count):

        logger.info(f"◎ Training on dataset #{i+1} ◎")

        if i < 1: # load base model
            arg2model = {
                "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
                "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "gemma2-9b": "google/gemma-2-9b-it",
                "qwen2-7b": "Qwen/Qwen2-7B-Instruct"
            }
            model_path = arg2model[args.model]        
        else: # Load last checkpoint
            last_output = output_dir + f"/{i}of{lr_scheduler_loader.max_count}"
            model_path = glob(f"{last_output}/checkpoint-*")[-1]

        global model
        model = get_model(args.peft_type, model_path, i+1)        
        if i > 0 and not args.save_per_dataset: # Remove the last checkpoint if save_per_dataset=False
            rmtree(last_output)

        logger.info("Setting dataset")
        train_dataset = train_dataset_list[i]
        eval_dataset = eval_dataset_list[i if len(eval_dataset_list) > 1 else 0] 

        logger.info("Setting optimizer and learning rate scheduler")
        optimizer, lr_scheduler = lr_scheduler_loader.get_optimizer_and_lr_scheduler(parameters=model.parameters())
        lr_init = lr_scheduler.lr_lambdas[0](0)
        logger.info(f"Initial learning rate of scheduler: {lr_init}")

        logger.info("Building SFTTrainer")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
            max_seq_length=args.max_sequence_length,
            dataset_num_proc=args.num_proc,
        )
        trainer.args.output_dir = output_dir + f"/{i+1}of{lr_scheduler_loader.max_count}"

        loop_start = time()
        trainer.train()
        loop_end = time()
        
        logger.debug(f"Time elapsed for training on dataset #{i+1}: {timedelta(seconds=round(loop_end - loop_start))}")

    total_end = time()
    logger.info("Training finished")
    logger.debug(f"Time elapsed for training in total: {timedelta(seconds=round(total_end - total_start))}")

    
    # Saving lr_scheduler plot

    plot_dir = f"{output_dir}/lr_scheduler_plot.jpg"
    logger.info(f"Saving lr_scheduler plot to: {plot_dir}")
    lr_scheduler_loader.save_lr_scheduler_plot(plot_dir)

    
    logger.info("Done")

    
if __name__ == "__main__":
    exit(main())
