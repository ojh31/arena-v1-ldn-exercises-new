#%%
import circuitsvis as cv
cv.examples.hello("Bob")
# %%
from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")
# %%
# import plotly.io as pio
# pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import repeat, rearrange, reduce
from fancy_einsum import einsum
import random
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked
from typing import List, Union, Optional, Tuple
from functools import partial
from tqdm import tqdm
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, display

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", caxis="", **kwargs) -> go.Figure:
    fig = px.imshow(
        utils.to_numpy(tensor), 
        color_continuous_midpoint=0.0, 
        color_continuous_scale="RdBu", 
        labels={"x":xaxis, "y":yaxis, "color":caxis}, 
        **kwargs
    )
    fig.show(renderer)
    return fig

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs) -> go.Figure:
    fig = px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)
    fig.show(renderer)
    return fig

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs) -> go.Figure:
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    fig.show(renderer)
    return fig
# %%
MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal", # defaults to "bidirectional"
    attn_only=True, # defaults to False

    tokenizer_name="EleutherAI/gpt-neox-20b", 
    # if setting from config, set tokenizer this way rather than passing it in explicitly
    # model initialises via AutoTokenizer.from_pretrained(tokenizer_name)

    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. use layernorm with weights and biases

    positional_embedding_type="shortformer" # this makes it so positional embeddings are used differently (makes induction heads cleaner to study)
)
# %%
WEIGHT_PATH = "./data/attn_only_2L_half.pth"

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)
    tokenizer = model.tokenizer
# %%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)
    tokens = tokens.to(device)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    model.reset_hooks()
#%%
def to_numpy(tensor):
    '''Helper function to convert things to numpy before plotting with Plotly.'''
    return tensor.detach().cpu().numpy()

def convert_tokens_to_string(tokens, batch_index=0):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]

seq_len = tokens.shape[-1]
n_components = model.cfg.n_layers * model.cfg.n_heads + 1

patch_typeguard()  # must call this before @typechecked

@typechecked
def logit_attribution(
    embed: TT["seq_len": seq_len, "d_model"],
    l1_results: TT["seq_len", "n_heads", "d_model"],
    l2_results: TT["seq_len", "n_heads", "d_model"],
    W_U: TT["d_model", "d_vocab"],
    tokens: TT["seq_len"],
) -> TT[seq_len-1, "n_components": n_components]:
    '''
    We have provided 'W_U_to_logits' which is a (d_model, seq_next) tensor where 
    each row is the unembed for the correct NEXT token at the current position.

    N.B. when searching for the correct next token we ignore
        * the first position token label
        * the model output at the last position.

    Inputs:
        embed: 
            the embeddings of the tokens (i.e. token + position embeddings)
            shape [s, nh]
        l1_results: 
            the outputs of the attention heads at layer 1 (with head as one of the dimensions)
            shape [s, n, nh]
        l2_results: 
            the outputs of the attention heads at layer 2 (with head as one of the dimensions)
            shape [s, n, nh]
        W_U: the unembedding matrix
            shape [nh, v]
        tokens:
            shape [s]
    Returns:
        Tensor representing the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
        shape [s-1, 2n + 1]
    '''
    W_U_to_logits = W_U[:, tokens[1:]] # shape [nh, s-1]
    direct_path = einsum(
        'dModel sN, sN dModel -> sN', W_U_to_logits, embed[:-1, :]
    ) # shape [s-1]
    direct_path = repeat(direct_path, 's -> s n', n=1)
    l1_path = einsum(
        'dModel sN, sN n dModel -> sN n', W_U_to_logits, l1_results[:-1, :, :]
    )
    l2_path = einsum(
        'dModel sN, sN n dModel -> sN n', W_U_to_logits, l2_results[:-1, :, :]
    )
    return t.cat((direct_path, l1_path, l2_path), dim=-1)
# %%
if MAIN:
    with t.inference_mode():
        batch_index = 0
        embed = cache["hook_embed"]
        l1_results = cache["result", 0] # same as cache["blocks.0.attn.hook_result"]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(
            embed, l1_results, l2_results, model.unembed.W_U, tokens[0]
        )
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where 
        # the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = (
            logits[batch_index, t.arange(len(tokens[0]) - 1), tokens[batch_index, 1:]]
        )
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-2, rtol=0)
# %%
def plot_logit_attribution(logit_attr: TT["seq", "path"], tokens: TT["seq"]):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [
        f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]
    imshow(
        to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", 
        caxis="logit", height=25*len(tokens)
    )

if MAIN:
    embed = cache["hook_embed"]
    l1_results = cache["blocks.0.attn.hook_result"]
    l2_results = cache["blocks.1.attn.hook_result"]
    logit_attr = logit_attribution(
        embed, l1_results, l2_results, model.unembed.W_U, tokens[0]
    )
    plot_logit_attribution(logit_attr, tokens)
# %%
if MAIN:
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
        display(html)
# %%
TOL = 0.3
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be 
    current-token heads
    '''
    heads = []
    for act_name, act_value in cache.items():
        if 'pattern' not in act_name:
            continue
        # act_value shape [n, s, s]
        layer = act_name.split('.')[1] # e.g. blocks.1.attn...
        for head, pattern in enumerate(act_value):
            currents = t.diag(pattern)
            # print(currents.sum().item() / pattern.sum().item())
            if currents.sum() > pattern.sum() * TOL:
                heads.append(f'{layer}.{head:d}')
    return heads


def prev_attn_detector(cache: ActivationCache):
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be 
    prev-token heads
    '''
    heads = []
    for act_name, act_value in cache.items():
        if 'pattern' not in act_name:
            continue
        # act_value shape [n, s, s]
        layer = act_name.split('.')[1] # e.g. blocks.1.attn...
        for head, pattern in enumerate(act_value):
            previous = t.diag(pattern, diagonal=-1)
            if previous.sum() > pattern.sum() * TOL:
                heads.append(f'{layer}.{head:d}')
    return heads

def first_attn_detector(cache: ActivationCache):
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be 
    first-token heads
    '''
    heads = []
    for act_name, act_value in cache.items():
        if 'pattern' not in act_name:
            continue
        # act_value shape [n, s, s]
        layer = act_name.split('.')[1] # e.g. blocks.1.attn...
        for head, pattern in enumerate(act_value):
            first = pattern[:, 0]
            if first.sum() > pattern.sum() * TOL:
                heads.append(f'{layer}.{head:d}')
    return heads

if MAIN:

    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
# %%
