# reference: transformers.optimization.py

import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from torch.optim import (
    ASGD,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    Adamax,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam
)
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda_linear(current_step, past_steps, min_lr_ratio, warmup_steps, total_steps):

    current_step += past_steps
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))

    progress = float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    factor = progress * (1 - min_lr_ratio) + min_lr_ratio
        
    return max(0.0, factor)


def lr_lambda_constant(current_step, past_steps, min_lr_ratio, warmup_steps, total_steps):

    current_step += past_steps
    if current_step < warmup_steps:
        return float(current_step) / float(max(1.0, warmup_steps))
        
    return 1.0


def lr_lambda_cosine(current_step, past_steps, min_lr_ratio, warmup_steps, total_steps):

    current_step += past_steps
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
        
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    factor = factor * (1 - min_lr_ratio) + min_lr_ratio
    
    return max(0.0, factor)


type2optimizer = {
    "asgd": ASGD,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adam": Adam,
    "adamw": AdamW,
    "adamax": Adamax,
    "lbfgs": LBFGS,
    "nadam": NAdam,
    "radam": RAdam,
    "rmsprop": RMSprop,
    "rprop": Rprop,
    "sgd": SGD,
    "sparseadam": SparseAdam
}

type2lambda = {
    "linear": lr_lambda_linear,
    "constant": lr_lambda_constant,
    "cosine": lr_lambda_cosine
}


class SplitLRSchedulerLoader():

    
    def __init__(
        self,
        steps_list: list,
        learning_rate: float = 5e-5,
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "linear",
        min_lr_ratio: float = 0.0,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        eps: float = 1e-8
    ):

        if optimizer_type not in type2optimizer.keys():
            raise ValueError(f"Invalid optimizer_type: {optimizer_type}. Please choose from {list(type2optimizer.keys())}.")

        if lr_scheduler_type not in type2lambda.keys():
            raise ValueError(f"Invalid lr_scheduler_type: {lr_scheduler_type}. Please choose from {list(type2lambda.keys())}.")

        if lr_scheduler_type == "constant" and min_lr_ratio != 0.0:
            raise ValueError(f"Constant lr_scheduler is not compatible with min_lr_ratio: {min_lr_ratio}. To use constant lr_scheduler, please set min_lr_ratio to 0.0.")

        if min_lr_ratio > 1.0:
            raise ValueError(f"Invalid min_lr_ratio: {min_lr_ratio}. Should no be bigger than 1.0.")

        self.steps_list = steps_list
        self.learning_rate = learning_rate        
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type
        self.min_lr_ratio = min_lr_ratio
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.eps = eps
        self.total_steps = sum(steps_list)
        self.max_count = len(steps_list)
        self.load_count = 0
        self.lr_schedulers = []

    
    def get_optimizer_and_lr_scheduler(self, parameters):

        if self.load_count >= self.max_count:
            raise CountError(f"Optimizer and lr_scheduler can be loaded up to {self.max_count} times. (current trial: {self.load_count + 1})")

        optimizer = type2optimizer[self.optimizer_type](
            params=parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.eps
        )
        
        lr_lambda = partial(
            type2lambda[self.lr_scheduler_type],
            min_lr_ratio=self.min_lr_ratio,
            past_steps=sum(self.steps_list[:self.load_count]),
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps
        )

        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,            
        )
        
        self.load_count += 1
        self.lr_schedulers.append(lr_scheduler)        

        return optimizer, lr_scheduler

        
    def save_lr_scheduler_plot(self, plot_dir):

        if self.load_count != self.max_count:
            raise CountError(f"Plot can be saved only after all lr_schedulers being loaded. (current progress: {self.load_count}/{self.max_count})")
    
        for i, steps in enumerate(self.steps_list):
            x = np.arange(sum(self.steps_list[:i]), sum(self.steps_list[:i+1]))
            y = np.array([self.lr_schedulers[i].lr_lambdas[0](k) * self.learning_rate for k in range(steps)])
            plt.plot(x, y, label=f"dataset {i}")
            
        plt.xlim(0, sum(self.steps_list))
        plt.ylim(0, self.learning_rate)        
        plt.legend()
        
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.save(plot_dir)
