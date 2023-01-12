import torch as t
from torch import nn
import numpy as np
import pandas as pd
from IPython.display import display
from typing import Callable

import w5d3_solutions


def test_positional_encoding(PosEnc: Callable):
    max_steps = 10
    embedding_dim = 100
    x = t.arange(max_steps)
    pos_enc = PosEnc(max_steps, embedding_dim)
    out = pos_enc(x)
    pos_enc_soln = w5d3_solutions.PositionalEncoding(max_steps, embedding_dim)
    out_soln = pos_enc_soln(x)
    assert out.shape == out_soln.shape
    t.testing.assert_allclose(out, out_soln)
    print('test_positional_encoding passed!')

def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

def compare_modules(actual: nn.Module, expected: nn.Module, name: str) -> None:
    param_list = sorted(
        [tuple(p_val.shape) for p_name, p_val in actual.named_parameters()], 
        key=lambda x: -t.prod(t.tensor(x)).item()
    )
    param_list_expected = sorted(
        [tuple(p_val.shape) for p_name, p_val in expected.named_parameters()], 
        key=lambda x: -t.prod(t.tensor(x)).item()
    )
    param_count = sum([p.numel() for p in actual.parameters() if p.ndim > 1])
    param_count_expected = sum([p.numel() for p in expected.parameters() if p.ndim > 1])
    error_msg = (
        f"Total number of (non-bias) parameters don't match: you have {param_count}, "
        f"expected number is {param_count_expected}; "
        f"found={param_list};  expected={param_list_expected}"
    )
    if param_count == param_count_expected:
        print(f"Parameter count test in {name} passed.")
    else:
        print_param_count(actual, expected)
        raise Exception(error_msg)

@t.inference_mode()
def test_groupnorm(GroupNorm, affine: bool):
    if not affine:
        x = t.arange(72, dtype=t.float32).view(3, 6, 2, 2)
        ref = t.nn.GroupNorm(num_groups=3, num_channels=6, affine=False)
        expected = ref(x)
        gn = GroupNorm(num_groups=3, num_channels=6, affine=False)
        actual = gn(x)
        t.testing.assert_close(actual, expected)
        print("All tests in `test_groupnorm(affine=False)` passed.")

    else:
        t.manual_seed(776)
        x = t.randn((3, 6, 8, 10), dtype=t.float32)
        ref = t.nn.GroupNorm(num_groups=3, num_channels=6, affine=True)
        ref.weight = nn.Parameter(t.randn_like(ref.weight))
        ref.bias = nn.Parameter(t.randn_like(ref.bias))
        expected = ref(x)
        gn = GroupNorm(num_groups=3, num_channels=6, affine=True)
        gn.weight.copy_(ref.weight)
        gn.bias.copy_(ref.bias)
        actual = gn(x)
        t.testing.assert_close(actual, expected)
        print("All tests in `test_groupnorm(affine=True)` passed.")

@t.inference_mode()
def test_self_attention(SelfAttention):
    channels = 16
    img = t.randn(1, channels, 64, 64)
    sa = SelfAttention(channels=channels, num_heads=4)
    out = sa(img)
    print("Testing shapes of output...")
    assert out.shape == img.shape
    print("Shape test in `test_self_attention` passed.")
    print("Testing values of output...")
    sa_solns = w5d3_solutions.SelfAttention(channels=channels, num_heads=4)
    try:
        sa.W_QKV = sa_solns.W_QKV
        sa.W_O = sa_solns.W_O
        out_actual = sa(img)
        out_expected = sa_solns(img)
        t.testing.assert_close(out_actual, out_expected)
        print("All tests in `test_self_attention` passed.")
    except:
        print(
            "Didn't find any linear layers called `W_QKV` and `W_O` with biases. "
            "Please change your linear layers to have these names, "
            "otherwise the values test can't be performed.")
    compare_modules(sa, sa_solns, 'test_self_attention')

@t.inference_mode()
def test_attention_block(AttentionBlock):
    ab = AttentionBlock(channels=16)
    ab_soln = w5d3_solutions.AttentionBlock(channels=16)
    img = t.randn(1, 16, 64, 64)
    out = ab(img)
    assert out.shape == img.shape
    print("Shape test in `test_attention_block` passed.")
    compare_modules(ab, ab_soln, 'test_attention_block')

@t.inference_mode()
def test_residual_block(ResidualBlock):
    in_channels = 6
    out_channels = 10
    step_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, in_channels, 32, 32)
    rb = ResidualBlock(in_channels, out_channels, step_dim, groups)
    out = rb(img, time_emb)
    print("Testing shapes of output...")
    assert out.shape == (1, out_channels, 32, 32)
    print("Shape test in `test_residual_block` passed.")
    print("Testing parameter count...")
    rb_soln = w5d3_solutions.ResidualBlock(in_channels, out_channels, step_dim, groups)
    compare_modules(rb, rb_soln, 'test_residual_block')

@t.inference_mode()
def test_downblock(DownBlock, downsample: bool):
    in_channels = 8
    out_channels = 12
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, in_channels, 32, 32)
    db = DownBlock(in_channels, out_channels, time_emb_dim, groups, downsample)
    out, skip = db(img, time_emb)
    print("Testing shapes of output...")
    assert skip.shape == (1, out_channels, 32, 32)
    if downsample:
        assert out.shape == (1, out_channels, 16, 16)
    else:
        assert out.shape == (1, out_channels, 32, 32)
    print("Shape test in `test_downblock` passed.")
    print("Testing parameter count...")
    db_soln = w5d3_solutions.DownBlock(in_channels, out_channels, time_emb_dim, groups, downsample)
    compare_modules(db, db_soln, 'test_downblock')

@t.inference_mode()
def test_midblock(MidBlock):
    mid_channels = 8
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, mid_channels, 32, 32)
    mid = MidBlock(mid_channels, time_emb_dim, groups)
    print("Testing shapes of output...")
    out = mid(img, time_emb)
    assert out.shape == (1, mid_channels, 32, 32)
    print("Shape test in `test_midblock` passed.")
    print("Testing parameter count...")
    mid_soln = w5d3_solutions.MidBlock(mid_channels, time_emb_dim, groups)
    param_count = sum([p.numel() for p in mid.parameters() if p.ndim > 1])
    param_count_expected = sum([p.numel() for p in mid_soln.parameters() if p.ndim > 1])
    assert param_count == param_count_expected, f"Total number of (non-bias) parameters don't match: you have {param_count}, expected number is {param_count_expected}."
    print("Parameter count test in `test_midblock` passed.\n")

@t.inference_mode()
def test_upblock(UpBlock, upsample):
    in_channels = 8
    out_channels = 12
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    # img = t.randn(1, out_channels, 16, 16)
    img = t.randn(1, in_channels, 16, 16)
    skip = t.rand_like(img)
    up = UpBlock(in_channels, out_channels, time_emb_dim, groups, upsample)
    out = up(img, time_emb, skip)
    # print("Testing shapes of output...")
    # if upsample:
    #     assert out.shape == (1, in_channels, 32, 32)
    # else:
    #     assert out.shape == (1, in_channels, 16, 16)
    # print("Shape test in `test_upblock` passed.")
    up_soln = w5d3_solutions.UpBlock(in_channels, out_channels, time_emb_dim, groups, upsample)
    print("Testing parameter count...")
    param_count = sum([p.numel() for p in up.parameters() if p.ndim > 1])
    param_count_expected = sum([p.numel() for p in up_soln.parameters() if p.ndim > 1])
    error_msg = f"Total number of (non-bias) parameters don't match: you have {param_count}, expected number is {param_count_expected}."
    if upsample==False:
        error_msg += "\nNote that upsample=False, so you don't need to define the convtranspose layer."
    assert param_count == param_count_expected, error_msg
    print("Parameter count test in `test_upblock` passed.\n")
    out_soln = up_soln(img, time_emb, skip)
    print('Testing upblock output shapes...')
    assert out.shape == out_soln.shape
    print('... shapes OK.')

@t.inference_mode()
def test_unet(Unet):
    # dim mults is limited by number of multiples of 2 in the image
    # 28 -> 14 -> 7 is ok but can't half again without having to deal with padding
    image_size = 28
    channels = 8
    batch_size = 8
    max_steps = 1_000
    config = w5d3_solutions.UnetConfig(
        image_shape=(8, 28, 28),
        channels=channels,
        dim_mults=(1, 2, 4),
        max_steps=max_steps,
    )
    model = Unet(config)
    x = t.randn((batch_size, channels, image_size, image_size))
    num_steps = t.randint(0, max_steps, (batch_size,))
    out = model(x, num_steps)
    assert out.shape == x.shape, (
        f'Unet out.shape={out.shape}, expected={x.shape}'
    )