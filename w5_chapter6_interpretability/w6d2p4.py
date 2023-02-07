#%%
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

from w6d2p3 import run_and_cache_model_repeated_tokens
# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

#%%
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

device = 'cpu' # t.device("cuda" if t.cuda.is_available() else "cpu")

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

    positional_embedding_type="shortformer", # this makes it so positional embeddings are used differently (makes induction heads cleaner to study)
    device=device,
)
# %%
WEIGHT_PATH = "./data/attn_only_2L_half.pth"

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)
    tokenizer = model.tokenizer
#%%
if MAIN:
    head_index = 4
    layer = 1
    wU = model.unembed.W_U
    wO = model.blocks[layer].attn.W_O[head_index]
    wV = model.blocks[layer].attn.W_V[head_index]
    wE = model.embed.W_E
    OV_circuit_full = (
        wE @ wV @ wO @ wU
    ) # replace with the matrix calculation W_U W_O W_V W_E
    
# %%
def to_numpy(tensor):
    '''Helper function to convert things to numpy before plotting with Plotly.'''
    return tensor.detach().cpu().numpy()

if MAIN:
    rand_indices = t.randperm(model.cfg.d_vocab)[:200]
    px.imshow(to_numpy(OV_circuit_full[rand_indices][:, rand_indices])).show()
# %%
def top_1_acc(OV_circuit_full):
    '''
    This should take the argmax of each column (ie over dim=0) and 
    return the fraction of the time that's equal to the correct logit
    '''
    return (
        OV_circuit_full.argmax(dim=0) == t.arange(0, OV_circuit_full.shape[0])
    ).sum().item() / len(OV_circuit_full)

if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc(OV_circuit_full))
# %%
if MAIN:
    try:
        del OV_circuit_full
    except:
        pass
    "YOUR CODE HERE, DEFINE OV_circuit_full_both"
    wO = model.blocks[layer].attn.W_O[head_index]
    OV_circuit_full_both = (
        wE @ 
        (
            model.blocks[1].attn.W_V[4] @ model.blocks[1].attn.W_O[4] +
            model.blocks[1].attn.W_V[10] @ model.blocks[1].attn.W_O[10]
        ) @ 
        wU
    )
    print("Top 1 accuracy for the full OV Circuit:", top_1_acc(OV_circuit_full_both))
    try:
        del OV_circuit_full_both
    except:
        pass
# %%
def mask_scores(
    attn_scores: TT["query_d_model", "key_d_model"]
):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

if MAIN:
    wP = model.pos_embed.W_pos
    wQ = model.blocks[0].attn.W_Q[7]
    wK = model.blocks[0].attn.W_K[7]
    pos_by_pos_pattern = mask_scores(
        wP @ wQ @ wK.T @ wP.T / (cfg.d_head ** 0.5)
    ).softmax(-1)

    imshow(to_numpy(pos_by_pos_pattern[:200, :200]), xaxis="Key", yaxis="Query")
# %%
def decompose_qk_input(cache: dict) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, position, d_model]
    '''
    heads = rearrange(cache["result", 0], 'b s h d -> b h s d').squeeze(0)
    decomposed_qk_input = t.cat(
        (cache['embed'], cache["pos_embed"], heads),
    ).to(device=device)
    return decomposed_qk_input


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head] 
    such that sum along axis 0 is just q
    '''
    wQ = model.blocks[1].attn.W_Q[ind_head_index]
    return einsum(
        'd_model d_head, h s d_model -> h s d_head', 
        wQ, 
        decomposed_qk_input,
    )


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head] 
    such that sum along axis 0 is just k
    exactly analogous as for q
    '''
    wK = model.blocks[1].attn.W_K[ind_head_index]
    return einsum(
        'd_model d_head, h s d_model -> h s d_head', 
        wK, 
        decomposed_qk_input,
    )


if MAIN:
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(
        model, seq_len, batch, device=device,
    )

    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    t.testing.assert_close(
        decomposed_qk_input.sum(0), 
        (rep_cache["resid_pre", 1][0] + rep_cache["pos_embed"][0]).to(device=device), 
        rtol=0.01, 
        atol=1e-05
    )
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(
        decomposed_q.sum(0), 
        rep_cache["blocks.1.attn.hook_q"][0, :, ind_head_index].to(device=device), 
        rtol=0.01, 
        atol=0.001
    )
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(
        decomposed_k.sum(0), 
        rep_cache["blocks.1.attn.hook_k"][0, :, ind_head_index].to(device=device), 
        rtol=0.01, 
        atol=0.01
    )
    component_labels = ["Embed", "PosEmbed"] + [f"L0H{h}" for h in range(model.cfg.n_heads)]
    imshow(
        to_numpy(decomposed_q.pow(2).sum([-1])), 
        xaxis="Pos", 
        yaxis="Component", 
        title="Norms of components of query"
    )
    imshow(
        to_numpy(decomposed_k.pow(2).sum([-1])),
        xaxis="Pos", 
        yaxis="Component", 
        title="Norms of components of key"
    )

#%%
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    decomposed_q: shape [2+num_heads, position, d_head] 
    decomposed_k: shape [2+num_heads, position, d_head] 

    return: shape [query_component, key_component, query_pos, key_pos]

    '''
    return einsum(
        "hQ seqQ headsize, hK seqK headsize -> "
        "hQ hK seqQ seqK", 
        decomposed_q, 
        decomposed_k
    ) / (decomposed_q.shape[-1] ** 0.5)


if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = reduce(
        decomposed_scores, 
        "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", 
        t.std,
    )
    # First plot: attention score contribution from 
    # (query_component, key_component) = (Embed, L0H7)
    imshow(
        to_numpy(t.tril(decomposed_scores[0, 9])), 
        title="Attention Scores for component from Q=Embed and K=Prev Token Head"
    )
    # Second plot: std dev over query and key positions, shown by component
    imshow(
        to_numpy(decomposed_stds), 
        xaxis="Key Component", 
        yaxis="Query Component", 
        title="Standard deviations of components of scores", 
        x=component_labels, 
        y=component_labels
    )
# %%
