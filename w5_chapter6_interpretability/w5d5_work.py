#%%
import functools
import json
import os
from typing import Any, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
from torch import nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange, repeat
import pandas as pd
import numpy as np
import random

import w5d5_tests
from w5d5_transformer import ParenTransformer, SimpleTokenizer

MAIN = __name__ == "__main__"
DEVICE = t.device("cpu")
# %%
if MAIN:
    model = ParenTransformer(
        ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3
    ).to(DEVICE)
    state_dict = t.load("w5d5_balanced_brackets_state_dict.pt")
    model.to(DEVICE)
    model.load_simple_transformer_state_dict(state_dict)
    model.eval()
    tokenizer = SimpleTokenizer("()")
    with open("w5d5_brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)

class DataSet:
    '''A dataset containing sequences, is_balanced labels, and tokenized sequences'''

    def __init__(self, data_tuples: list):
        '''
        data_tuples is List[Tuple[str, bool]] signifying sequence and label
        '''
        self.strs = [x[0] for x in data_tuples]
        self.isbal = t.tensor([x[1] for x in data_tuples]).to(device=DEVICE, dtype=t.bool)
        self.toks = tokenizer.tokenize(self.strs).to(DEVICE)
        self.open_proportion = t.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = t.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self) -> int:
        return len(self.strs)

    def __getitem__(self, idx) -> Union["DataSet", tuple[str, t.Tensor, t.Tensor]]:
        if type(idx) == slice:
            return self.__class__(list(zip(self.strs[idx], self.isbal[idx])))
        return (self.strs[idx], self.isbal[idx], self.toks[idx])

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)

    @classmethod
    def with_length(cls, data_tuples: list[tuple[str, bool]], selected_len: int) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if len(s) == selected_len])

    @classmethod
    def with_start_char(cls, data_tuples: list[tuple[str, bool]], start_char: str) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if s[0] == start_char])

if MAIN:
    N_SAMPLES = 5000
    data_tuples = data_tuples[:N_SAMPLES]
    data = DataSet(data_tuples)
    print('examples: ', random.sample(data.strs, 5))
    print('isbal: ', pd.Series(data.isbal).value_counts())
    char_lens = [len(s) for s in data.strs]
    print('len: ', pd.Series(char_lens).value_counts().head())
    fig = px.histogram(x=char_lens)
    fig.update_layout(dict(
        xaxis_title='string length'
    ))
    fig.show()

# %%
def is_balanced_forloop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    level = 0
    for c in parens:
        if c == '(':
            level += 1
        elif c == ')':
            level -= 1
        else:
            raise ValueError(f'Bad character in parens str: {c}')
        if level < 0:
            return False
    return level == 0

if MAIN:
    examples = [
        "()", 
        "))()()()()())()(())(()))(()(()(()(", 
        "((()()()()))", 
        "(()()()(()(())()", 
        "()(()(((())())()))"
    ]
    labels = [True, False, True, False, True]
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")
# %%
def is_balanced_vectorized(tokens: t.Tensor) -> bool:
    '''
    tokens: 
        sequence of tokens including begin, end and pad tokens 
        - recall that 3 is '(' and 4 is ')'
    '''
    level_diffs = t.where(
        tokens == 3,
        1,
        t.where(tokens == 4, -1, 0)
    )
    levels = level_diffs.cumsum(dim=-1)
    return (levels >= 0).all() and levels[-1] == 0

if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")
# %%
if MAIN:
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    prob_balanced = out.exp()[:, 1]
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))

def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    '''Return probability that each example is balanced'''
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out

if MAIN:
    test_set = data
    n_correct = t.sum((run_model_on_data(model, test_set).argmax(-1) == test_set.isbal).int())
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
# %%
def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    '''
    Use the weights of the final linear layer (model.decoder) to 
    identify the direction in the space that goes into the linear layer (and out of the LN) 
    corresponding to an 'unbalanced' classification. 
    Hint: this is a one line function.

    out: shape (e=56, )
    '''
    return (model.decoder.weight[0, :] - model.decoder.weight[1, :])
# %%
def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    inputs = []
    def hook(hook_module: nn.Module, input: Tuple, output: t.Tensor) -> None:
        assert hook_module == module
        for inp in input:
            inputs.append(inp)

    handle = module.register_forward_hook(hook)
    run_model_on_data(model, data)
    handle.remove()
    return t.cat(inputs)

def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    outputs = []
    def hook(hook_module: nn.Module, input: Tuple, output: t.Tensor) -> None:
        assert hook_module == module
        outputs.append(output)

    handle = module.register_forward_hook(hook)
    run_model_on_data(model, data)
    handle.remove()
    return t.cat(outputs)

if MAIN:
    w5d5_tests.test_get_inputs(get_inputs, model, data)
    w5d5_tests.test_get_outputs(get_outputs, model, data)
# %%
def get_ln_fit(
    model: ParenTransformer, data: DataSet, ln_module: nn.LayerNorm, seq_pos: Union[None, int]
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit for all sequence positions. 
    Otherwise, fit only for given seq_pos.

    Returns: A tuple of 
        - a (fitted) sklearn LinearRegression object
        - a dimensionless tensor containing the r^2 of the fit (hint: 
            wrap a value in torch.tensor() to make a dimensionless tensor)
    '''
    inputs = get_inputs(model, data, ln_module)
    outputs = get_outputs(model, data, ln_module)
    if seq_pos is not None:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]
    lr = LinearRegression()
    lr.fit(inputs, outputs)
    r2 = t.tensor(lr.score(inputs, outputs))
    return (lr, r2)


if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, model.norm, seq_pos=0)
    print("r^2: ", r2)
    w5d5_tests.test_final_ln_fit(model, data, get_ln_fit)
# %%
def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    x_norm -> (ln) x_linear -> (linear and softmax) logit_diff
    logit_diff is maximised by post_final_ln_dir: shape (e=56, )
    post_final_ln_dir is maximised by ... in seq_pos=0

    pre_final_ln_dir: shape (e, ) = (56, )
    L: shape (e, e) = (56, 56)
    post_final_ln_dir: shape (e, ) = (56, )
    '''
    post_final_ln_dir = get_post_final_ln_dir(model)
    ln_fit, _ = get_ln_fit(model, data, model.norm, seq_pos=0)
    L = t.tensor(ln_fit.coef_)
    pre_final_ln_dir = post_final_ln_dir @ L
    return pre_final_ln_dir


if MAIN:
    w5d5_tests.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)
# %%
def get_out_by_head(
    model: ParenTransformer, data: DataSet, layer: int
) -> t.Tensor:
    '''
    Get the output of the heads in a particular layer when the model is run on the data.
    Returns a tensor of shape (batch, num_heads, seq, embed_width)
    '''
    W_O = model.layers[layer].self_attn.W_O
    r = get_inputs(model, data, W_O)
    r_block = rearrange(
        r, 
        'b s (n h)-> b n s h', 
        n=model.nhead,
    ) # split up the rows of r
    W_block = rearrange(
        W_O.weight, 
        'e (n h) -> n e h', 
        n=model.nhead,
    ) # split up the columns of W_O
    out = einsum(
        'n e h, b n s h-> b n s e', 
        W_block, 
        r_block,
    ) # The common dim in the matmuls here is h
    return out

if MAIN:
    w5d5_tests.test_get_out_by_head(get_out_by_head, model, data)
# %%
def get_out_by_components(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing 
    the output of the model's components when run on the data.

    The first dimension is [
        embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2
    ]
    '''
    embeddings = get_outputs(model, data, model.pos_encoder).unsqueeze(dim=1)
    output = [embeddings]
    for i, layer in enumerate(model.layers):
        output.append(get_out_by_head(model, data, i))
        output.append(get_outputs(model, data, layer.linear2).unsqueeze(dim=1))
    o_tensor = t.cat(output, dim=1)
    o_tensor = rearrange(o_tensor, 'b c s e -> c b s e')
    return o_tensor

if MAIN:
    w5d5_tests.test_get_out_by_component(get_out_by_components, model, data)
# %%
