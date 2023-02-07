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

# seq_len = tokens.shape[-1]
# n_components = model.cfg.n_layers * model.cfg.n_heads + 1

patch_typeguard()  # must call this before @typechecked

@typechecked
def logit_attribution(
    embed: TT["seq_len", "d_model"],
    l1_results: TT["seq_len", "n_heads", "d_model"],
    l2_results: TT["seq_len", "n_heads", "d_model"],
    W_U: TT["d_model", "d_vocab"],
    tokens: TT["seq_len"],
) -> TT["seq_len_minus1", "n_components"]:
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
def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def head_ablation(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_no: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    attn_result[:, :, head_no, :] = 0.
    return attn_result

def get_ablation_scores(
    model: HookedTransformer, 
    tokens: TT["batch", "seq"]
) -> TT["n_layers", "n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in 
    cross entropy loss from ablating the output of each head.
    '''
    logits = model(tokens, return_type='logits')
    base_loss = cross_entropy_loss(logits, tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    out = t.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            hook = partial(head_ablation, head_no=head)
            ablated_logits = model.run_with_hooks(
                tokens, 
                fwd_hooks=[(
                    utils.get_act_name("result", layer), 
                    hook
                )],
                return_type='logits', 
            )
            ablated_loss = cross_entropy_loss(ablated_logits, tokens)
            out[layer, head] = ablated_loss - base_loss
    return out

if MAIN:
    ablation_scores = get_ablation_scores(model, tokens)
    imshow(
        ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", 
        title="Logit Difference After Ablating Heads", text_auto=".2f"
    )
# %%
def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, 
    seq_len: int, 
    batch: int = 1,
    device: str = None,
) -> tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning 
    logits, tokens and cache

    Add a prefix token, since the model was always trained to have one.

    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: The cache of the model run on rep_tokens
    '''
    prefix = t.ones((batch, 1), dtype=t.int64, device=device) * model.tokenizer.bos_token_id
    noise = t.randint(low=0, high=model.cfg.d_vocab, size=(batch, seq_len, ), device=device)
    rep_tokens = t.cat((prefix, noise, noise), dim=-1)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_logits, rep_tokens, rep_cache

def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]

if MAIN:
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(
        model, seq_len, batch, device=device,
    )
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print(f"Performance on the first half: {ptl[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {ptl[seq_len:].mean():.3f}")
    fig = px.line(
        to_numpy(ptl), hover_name=rep_str[1:],
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
        labels={"index": "Sequence position", "value": "Loss"}
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=49.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=49.5, x1=99, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()
# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be 
    induction heads

    Remember:
        The tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    _, out_len, _ = cache['blocks.0.hook_resid_post'].shape
    seq_len = out_len // 2
    heads = []
    for act_name, act_value in cache.items():
        if 'pattern' not in act_name:
            continue
        # act_value shape [n, s, s]
        layer = act_name.split('.')[1] # e.g. blocks.1.attn...
        for head, pattern in enumerate(act_value[0]):
            previous = t.diag(pattern, diagonal=-(seq_len - 1))
            if previous.sum() > pattern.sum() * TOL:
                heads.append(f'{layer}.{head:d}')
    return heads

if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%
if MAIN:
    batch, full_len, d_model = embed.shape
    assert batch == 1
    seq_len = full_len // 2
    embed = rep_cache["hook_embed"]
    l1_results = rep_cache["blocks.0.attn.hook_result"]
    l2_results = rep_cache["blocks.1.attn.hook_result"]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]
    first_half_logit_attr = logit_attribution(
        embed[0, :seq_len + 1, :], 
        l1_results[0, :seq_len + 1, :, :], 
        l2_results[0, :seq_len + 1, :, :], 
        model.unembed.W_U, 
        first_half_tokens
    )
    second_half_logit_attr = logit_attribution(
        embed[0, seq_len:, :], 
        l1_results[0, seq_len:, :, :], 
        l2_results[0, seq_len:, :, :], 
        model.unembed.W_U, 
        second_half_tokens,
    )
    plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    plot_logit_attribution(second_half_logit_attr, second_half_tokens)
# %%
if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    imshow(
        ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads (detecting induction heads)", text_auto=".2f"
    )
# %%
