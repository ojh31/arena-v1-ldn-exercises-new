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
#%% [markdown]
#### Part 1: Introduction
# %%
device = "cuda" if t.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
print('device:', device)
# %%
model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. 
You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. 
See [model_details.md](TODO: link) for a description of all supported models. 
Each model is loaded into the consistent HookedTransformer architecture, 
designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. 
To try the model the model out, let's find the loss on this paragraph!'''
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = model.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
# %%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0, "attn"]
print(attention_pattern.shape)
gpt2_str_tokens = model.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_heads(tokens=gpt2_str_tokens, attention=attention_pattern)
# %%
layer_to_ablate = 0
head_index_to_ablate = 8

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader
def head_ablation_hook(
    value: TT["batch", "pos", "head_index", "d_head"],
    hook: HookPoint
) -> TT["batch", "pos", "head_index", "d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

original_loss = model(gpt2_tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    gpt2_tokens, 
    return_type="loss", 
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook
    )]
)
print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")
# %%
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
    # model.to_single_token maps str -> token index
    # If the string is not a single token, it raises an error.
    correct_index = model.to_single_token(correct_answer)
    incorrect_index = model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# We run on the clean prompt with the cache so we store activations to patch in later.
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logit_diff = logits_to_logit_diff(clean_logits)
print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

# We don't need to cache on the corrupted prompt.
corrupted_logits = model(corrupted_tokens)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
# %%
def residual_stream_patching_hook(
    resid_pre: TT["batch", "pos", "d_model"],
    hook: HookPoint,
    position: int
) -> TT["batch", "pos", "d_model"]:
    '''
    We choose to act on the residual stream at the start of the layer, 
    so we call it resid_pre.

    Each HookPoint has a name attribute giving the name of the hook.
    '''
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

# We make a tensor to store the results for each patching run. 
# We put it on the model's device to avoid needing to move things between the GPU and CPU, 
# which can be slow.
num_positions = len(clean_tokens[0])
ioi_patching_result = t.zeros(
    (model.cfg.n_layers, num_positions), device=model.cfg.device
)

for layer in tqdm(range(model.cfg.n_layers)):
    for position in range(num_positions):
        # Use functools.partial to create a temporary hook function with the position fixed
        temp_hook_fn = partial(residual_stream_patching_hook, position=position)
        # Run the model with the patching hook
        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        ]) # 'blocks.0.hook_resid_pre'
        # Calculate the logit difference
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        # Store the result, normalizing by the clean and corrupted logit difference so 
        # it's between 0 and 1 (ish)
        ioi_patching_result[layer, position] = (
            (patched_logit_diff - corrupted_logit_diff) /
            (clean_logit_diff - corrupted_logit_diff)
        )
# %%
# Add the index to the end of the label, because plotly doesn't like duplicate labels
token_labels = [
    f"{token}_{index}" 
    for index, token in enumerate(model.to_str_tokens(clean_tokens))
]
fig = imshow(
    ioi_patching_result, 
    x=token_labels, 
    xaxis="Position", 
    yaxis="Layer", 
    title="Normalized Logit Difference After Patching Residual Stream on the IOI Task"
)
fig.show(None)
# %%
batch_size = 10
seq_len = 50
random_tokens = t.randint(1000, 10000, (batch_size, seq_len)).to(model.cfg.device)
repeated_tokens = repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")
repeated_logits = model(repeated_tokens)
correct_log_probs = model.loss_fn(repeated_logits, repeated_tokens, per_token=True)
loss_by_position = reduce(correct_log_probs, "batch position -> position", "mean")
fig = line(
    loss_by_position, xaxis="Position", yaxis="Loss", 
    title="Loss by position on random repeated tokens"
)
#%%
# We make a tensor to store the induction score for each head. 
# We put it on the model's device to avoid needing to move things between the GPU and CPU, 
# which can be slow.
induction_score_store = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), 
    device=model.cfg.device
)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to 
    # source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(
        dim1=-2, dim2=-1, offset=1-seq_len
    ) # diagonal is dim2 == dim1 + offset
    # Get an average score per head
    induction_score = reduce(
        induction_stripe, 
        "batch head_index position -> head_index", 
        "mean",
    )
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

model.run_with_hooks(
    repeated_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

fig = imshow(
    induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head"
)
# %%
induction_head_layer = 5
induction_head_index = 5
single_random_sequence = t.randint(1000, 10000, (1, 20)).to(model.cfg.device)
repeated_random_sequence = repeat(
    single_random_sequence, "batch seq_len -> batch (2 seq_len)"
)
def visualize_pattern_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    display(
        cv.attention.attention_heads(
            tokens=model.to_str_tokens(repeated_random_sequence), 
            attention=pattern[
                0, induction_head_index, :, :
            ][None, :, :] # Add a dummy axis, as CircuitsVis expects 3D patterns.
        )
    )

model.run_with_hooks(
    repeated_random_sequence, 
    return_type=None, 
    fwd_hooks=[(
        utils.get_act_name("pattern", induction_head_layer), 
        visualize_pattern_hook
    )]
)
# %%
distilgpt2 = HookedTransformer.from_pretrained("distilgpt2")
# We make a tensor to store the induction score for each head. 
# We put it on the model's device to avoid needing to move things between 
# the GPU and CPU, which can be slow.
distilgpt2_induction_score_store = t.zeros(
    (distilgpt2.cfg.n_layers, distilgpt2.cfg.n_heads), 
    device=distilgpt2.cfg.device
)
def induction_score_hook(
    pattern: TT["batch", "head_index", "dest_pos", "source_pos"],
    hook: HookPoint,
):
    # We take the diagonal of attention paid from each destination position to 
    # source positions seq_len-1 tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = reduce(
        induction_stripe, 
        "batch head_index position -> head_index", 
        "mean"
    )
    # Store the result.
    distilgpt2_induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on 
# attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

distilgpt2.run_with_hooks(
    repeated_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

fig = imshow(
    distilgpt2_induction_score_store, xaxis="Head", yaxis="Layer", 
    title="Induction Score by Head in Distil GPT-2"
)
# %%
for name, param in model.named_parameters():
    if name.startswith("blocks.0."):
        print(name, param.shape)
# %%
for name, param in model.named_parameters():
    if not name.startswith("blocks"):
        print(name, param.shape)
# %%
test_prompt = "The quick brown fox jumped over the lazy dog"
print("Num tokens:", len(model.to_tokens(test_prompt)))

def print_name_shape_hook_function(activation, hook):
    print(hook.name, activation.shape)


def not_in_late_block_filter(name: str):
    return name.startswith("blocks.0.") or not name.startswith("blocks")

model.run_with_hooks(
    test_prompt,
    return_type=None,
    fwd_hooks=[(not_in_late_block_filter, print_name_shape_hook_function)],
)
# %%
unembed_bias = model.unembed.b_U
bias_values, bias_indices = unembed_bias.sort(descending=True)
# %%
top_k = 20
print(f"Top {top_k} values")
for i in range(top_k):
    print(f"{bias_values[i].item():.2f} {repr(model.to_string(bias_indices[i]))}")

print("...")
print(f"Bottom {top_k} values")
for i in range(top_k, 0, -1):
    print(f"{bias_values[-i].item():.2f} {repr(model.to_string(bias_indices[-i]))}")
# %%
john_bias = model.unembed.b_U[model.to_single_token(' John')]
mary_bias = model.unembed.b_U[model.to_single_token(' Mary')]

print(f"John bias: {john_bias.item():.4f}")
print(f"Mary bias: {mary_bias.item():.4f}")
print(f"Prob ratio bias: {t.exp(john_bias - mary_bias).item():.4f}x")
#%%
