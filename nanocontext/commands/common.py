import math
import time

import click
import numpy as np
import torch

from nanocontext.data.broadcast_tree import (
    broadcast_tree_data_loader, SimpleEngine, StatefulEngine, save_engine, load_engine,
    SummaryTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer,
    PerfectTreeTokenizer
)
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.evaluate.coloring import check_validity, get_root_constraint, UnsatisfiedException, evaluate_entropy
from nanocontext.evaluate.ising import evaluate_moments, gather_magnets, compute_moments, evaluate_perplexity
from nanocontext.train import NanochatTrainerConfig, NanochatTrainer, TrainerSignal
from nanocontext.sample import NanochatSampler
from nanocontext.tree import IsingBroadcastChannel, ColoringBroadcastChannel, PerfectTreeConfig, InferenceTree
from nanocontext.tree.broadcast import markov_forest
from nanocontext.tree.coloring import ColoringSpace
from nanocontext.tree.ising import IsingSpace
from nanocontext.utils import (
    ddp_context, ddp_world_size, device_to_use, is_main_process, compute_moments,
    main_process, synchronize, RNGManager
)

import wandb

from nanocontext.utils.dist import main_process


echo = main_process(click.echo)


def make_prompt(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig):
    if isinstance(tokenizer, SummaryTokenizer):
        return tokenizer.init_summary_tokens(config)
    return [tokenizer.bos_token]


def get_max_tokens(d, height):
    return (d ** (height - 1)) * (d + 1) - 1


def display_recon_stat(stat):
    echo(f"Unsatisfied: {stat['unsatisfied']['total']}")
    for depth in range(len(stat["unsatisfied"]["details"])):
        echo(f"  At depth {depth}: {stat['unsatisfied']['details'][depth]}")
    echo(f"Invalid: {stat['invalid']}")
    echo(f"Constrained: {stat['constrained']}")
    echo(f"Free: {stat['free']}")
