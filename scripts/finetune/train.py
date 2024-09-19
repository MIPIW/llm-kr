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

import wandb
import torch
from trl import SFTConfig
from trl import SFTTrainer
from peft import PeftModel
from peft import LoraConfig
from peft import AdaLoraConfig
from peft import get_peft_model
from datasets import load_dataset
##########################################ADDED###############################################
from transformers.trainer import *
##########################################ADDED###############################################

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

m.add_argument("--model", type=str, required=True, choices=["llama3-8b", "llama3.1-8b", "gemma2-9b", "qwen2-7b", "exaone3-7.8b"], help="IT model type (required)")
m.add_argument("--train_dataset", type=str, nargs="+", default=["/node_storage2/data_llm_kr/data_it_rpd_train_240806.csv", "/node_storage2/data_llm_kr/data_it_qa_train_240806.csv"], help="train datasets, could be more than 1 (should be in format of csv)")
m.add_argument("--eval_dataset", type=str, nargs="+", default=["/node_storage2/data_llm_kr/data_it_rpd_eval_240806.csv", "/node_storage2/data_llm_kr/data_it_qa_eval_240806.csv"], help="eval datasets, could be more than 1 (should be in format of csv)")
m.add_argument("--task_templates", type=str, default="/node_storage2/data_llm_kr/instruction_template_240806.json", help="task template")
m.add_argument("--output_dir", type=str, default="./outputs/checkpoints", help="output directory for model checkpoint")

h.add_argument("--train_batch_size", type=int, default=4, help="train batch size")
h.add_argument("--eval_batch_size", type=int, default=4, help="eval batch size")
h.add_argument("--epochs", type=int, default=1, help="# of train epochs")
h.add_argument("--gradient_accumulation_steps", type=int, default=128, help="gradient accumulation steps")
h.add_argument("--max_sequence_length", type=int, default=1024, help="max sequence length")

p.add_argument("--peft_type", type=str, default="no", choices=["no", "lora", "dora", "adalora"], help="whether to use LoRA or DoRA; defaults to 'no'")
p.add_argument("--lora_rank", type=int, default=8, help="rank for LoRA")
p.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha for LoRA")
p.add_argument("--lora_dropout", type=float, default=0.1, help="dropout probability for LoRA")
p.add_argument("--lora_target", type=str, default="no_qk", choices=["all", "no_qk", "no_v", "no_qkv"], help="target modules to apply LoRA; defaults to 'all'")
p.add_argument("--adalora_init_rank", type=int, default=16, help="initial rank for AdaLoRA")

l.add_argument("--learning_rate", type=float, default=4.5e-6, help="learning rate")
l.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "linear", "cosine"], help="learning rate scheduler type; defaults to 'cosine'")
l.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of learning rate warmup steps to total steps")
l.add_argument("--warmup_steps", type=int, default=0, help="learning rate warmup steps, overwrites warmup ratio if set")
l.add_argument("--min_lr_ratio", type=float, default=0.01, help="ratio of minimum learning rate to original learning rate")
l.add_argument("--min_lr", type=float, default=0.0, help="minimum learning rate, overwrites minimun learning rate ratio if set")
l.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for AdamW optimizer")
l.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon(ε) for AdamW optimizer")

s.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="save strategy")
s.add_argument("--save_steps", type=int, default=500, help="save steps")
s.add_argument("--save_total_limit", type=int, default=1, help="# of last checkpoints to save")
s.add_argument("--load_best_model_at_end", action="store_true", help="whether to save best checkpoint")
s.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="metric for best checkpoint")

e.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"], help="eval strategy")
e.add_argument("--eval_steps", type=int, default=500, help="eval steps")
e.add_argument("--eval_accumulation_steps", type=int, default=None, help="eval accumulation steps")

o.add_argument("--logging_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="logging strategy")
o.add_argument("--logging_steps", type=int, default=1, help="logging steps")
o.add_argument("--report_to_wandb", action="store_true", help="whether to report logs to wandb, overwrite logging settings if set")
o.add_argument("--num_proc", type=int, default=16, help="# of processors to be used")
o.add_argument("--seed", type=int, default=42, help="random seed for random, numpy, torch")

args = parser.parse_args()


### Setting Logger  

output_dir = f"{args.output_dir.rstrip("/")}/{args.model}/{args.peft_type}"
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


def set_peft_config():
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_target == "no_qkv":
        target_modules = target_modules[3:]
    elif args.lora_target == "no_qk":
        target_modules = target_modules[2:]
    elif args.lora_target == "no_v":
        target_modules.remove("v_proj")
    else: # args.lora_target == all
        pass
    
    peft_config_dict = {
        "use_dora": True if args.peft_type == "dora" else False,
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules
    }
    if args.peft_type == "adalora":
        peft_config_dict["init_r"] = args.adalora_init_rank
        peft_config = AdaLoraConfig(
            peft_type = "ADALORA",
            task_type="CAUSAL_LM",
            **peft_config_dict
        )
    else: 
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            **peft_config_dict
        )
    
    logger.debug("〓〓〓〓〓 PEFT Configuration 〓〓〓〓〓")
    logger.debug(f"- peft type: {args.peft_type}")
    for k, v in peft_config_dict.items():
        logger.debug(f"- {k}: {v}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")
    
    return peft_config


def get_tokenizer(model_path):

    logger.info(f"Loading tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"}) # Llama3 doesn't have pad_token

    return tokenizer


def get_model(model_path, adapter_path, load_count):

    logger.info(f"Loading base model from: {model_path}")
    
    # no flash attention 2 only for Gemma
    attn_implementation = "eager" if model_path == "google/gemma-2-9b-it" else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation
    )
    model.resize_token_embeddings(len(tokenizer))

    if load_count < 2:        
        if not args.peft_type == "no":
            logger.info("Setting PEFT configuration")
            peft_config = set_peft_config()
            logger.info("Building PEFT model")
            model = get_peft_model(model, peft_config=peft_config) 
            
            logger.debug("〓〓〓〓〓〓 Model Information 〓〓〓〓〓〓")
            logger.debug(f"- model name or path: {model_path}")
            logger.debug(f"- peft type: {args.peft_type}")    
            for message in get_trainable_parameters(model):
                logger.debug(message)
            logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")   
            
    else:        
        if not args.peft_type == "no":
            logger.info("Building PEFT model")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    
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
        dataloader_drop_last=False,
        disable_tqdm=False
    )

    logger.debug("〓〓〓〓〓 SFT  Configuration 〓〓〓〓〓")
    for k, v in sft_config_dict.items():
        logger.debug(f"- {k}: {v}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")
    
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
    logger.debug(f"- minimum learning rate ratio: {lr_scheduler_loader.min_lr_ratio}")    
    logger.debug(f"- # of lr_schedulers: {lr_scheduler_loader.max_count}")
    logger.debug(f"- # of steps in total: {lr_scheduler_loader.total_steps}")
    logger.debug(f"- # of warmup steps: {lr_scheduler_loader.warmup_steps}")
    logger.debug(f"- weight decay: {lr_scheduler_loader.weight_decay}")
    logger.debug(f"- adam epsilon: {lr_scheduler_loader.eps}")
    logger.debug("〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓〓")

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
        <|endoftext|>""",
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": """
        [|system|]당신의 역할은 한국어로 답변하는 **한국어 AI 어시트턴트**입니다. 주어진 질문에 대해 한국어로 답변해주세요.[|endofturn|]
        [|user|]아래 질문을 한국어로 정확하게 답변해주세요. **질문**: {}[|endofturn|]
        [|assistant|]{}[|endofturn|]
        """
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


def main():    
    
    ### Setting random seed
    
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
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
        "exaone3-7.8b": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
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

        if i < 1:
            adapter_path = ""
        else: # Load last checkpoint
            last_output = output_dir + f"/{i}of{lr_scheduler_loader.max_count}"
            adapter_path = glob(f"{last_output}/checkpoint-*")[-1]

        global model
        model = get_model(model_path, adapter_path, i+1)        
        if args.peft_type == "adalora":
            model.peft_config["default"].total_step = steps_list[i]
            
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
