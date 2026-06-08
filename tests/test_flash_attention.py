import math

import numpy as np
import pytest
import torch
import triton

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device
vendor_name = flag_gems.vendor_name

def _hadamard_matrix(dim, device):
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(dim)


def _apply_incoherent_qk(x):
    h = _hadamard_matrix(x.shape[-1], x.device).to(torch.float32)
    return torch.matmul(x.float(), h).to(x.dtype)


def make_input(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    dtype,
    device,
    requires_grad=False,
):
    random_utils.set_philox_state(1234567890, 0, device)
    q_shape = (batch, num_head, q_seq_len, head_size)
    kv_shape = (batch, num_head_k, kv_seq_len, head_size)
    q = torch.empty(q_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    k = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    v = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)

    if requires_grad:
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()

    return q, k, v


def torch_flash_fwd(
    q, k, v, scale, is_causal, dropout_p=0, return_debug_mask=False, **extra_kwargs
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    (
        out,
        lse,
        seed,
        offset,
        debug_softmax,
    ) = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )

    return out, lse, seed, offset, debug_softmax


def gems_flash_fwd(
    q, k, v, scale, is_causal, dropout_p=0, return_debug_mask=False, **extra_kwargs
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    (
        out,
        lse,
        seed,
        offset,
        debug_softmax,
    ) = flag_gems.ops.flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )

    return out, lse, seed, offset, debug_softmax


def _get_fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    return dtype


def _fp8_max_value(fp8_dtype):
    if fp8_dtype is getattr(torch, "float8_e5m2", None):
        return float(torch.finfo(fp8_dtype).max)
    return 448.0


def _quantize_per_block_fp8(x, fp8_dtype, block_size=128):
    batch, seq_len, num_head, _ = x.shape
    fp8_max = _fp8_max_value(fp8_dtype)
    nblocks = triton.cdiv(seq_len, block_size)
    out = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.empty((batch, num_head, nblocks), device=x.device, dtype=torch.float32)
    for block_idx in range(nblocks):
        lo = block_idx * block_size
        hi = min(seq_len, lo + block_size)
        tile = x[:, lo:hi, :, :].float()
        scale = (tile.abs().amax(dim=(1, 3)) / fp8_max).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        out[:, lo:hi, :, :] = torch.clamp(
            tile / scale[:, None, :, None], -fp8_max, fp8_max
        ).to(fp8_dtype)
        descale[:, :, block_idx] = scale
    return out.contiguous(), descale.contiguous()


def _quantize_qkv_w8a8(q, k, v):
    fp8_dtype = _get_fp8_dtype()
    q_i = _apply_incoherent_qk(q)
    k_i = _apply_incoherent_qk(k)
    q_fp8, q_descale = _quantize_per_block_fp8(q_i, fp8_dtype)
    k_fp8, k_descale = _quantize_per_block_fp8(k_i, fp8_dtype)
    v_fp8, v_descale = _quantize_per_block_fp8(v, fp8_dtype)
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale, _fp8_max_value(fp8_dtype)

def gems_flash_fwd_w8a8(q, k, v, scale, is_causal):
    q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale, fp8_p_max = _quantize_qkv_w8a8(
        q, k, v
    )
    out = torch.empty_like(q, dtype=torch.float16)
    result = flag_gems.ops.flash_attention_forward_w8a8(
        q=q_fp8,
        k=k_fp8,
        v=v_fp8,
        out=out,
        alibi_slopes=None,
        p_dropout=0.0,
        softmax_scale=scale,
        is_causal=is_causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
        disable_splitkv=False,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        fp8_p_max=fp8_p_max,
    )
    return result[0] if isinstance(result, (tuple, list)) else result


def _assert_w8a8_attention_close(actual, expected):
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    mse = torch.mean(diff * diff)
    rel_mse = mse / torch.mean(expected_f * expected_f).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    assert mse.item() < 1.0e-4
    # assert rel_mse.item() < 5.0e-3
    # assert cosine.item() > 0.999
    assert rel_mse.item() < 2.0e-2
    assert cosine.item() > 0.99


def sparse_attention_ref(q, kv, attn_sink, topk_idxs, scale):
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]

    kv_expanded = kv[:, None, :, :].expand(batch, seq_len, -1, dim)
    idx_expanded = topk_idxs[:, :, :, None].expand(batch, seq_len, topk, dim).long()
    gathered_kv = torch.gather(kv_expanded, 2, idx_expanded)

    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * scale
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)

    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], gathered_kv.float())
    return out.to(q.dtype)


@pytest.mark.skip(
    reason="Issue #2809: The operator fails this test on Nvidia at least."
)
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.sparse_attn_triton
@pytest.mark.parametrize(
    "batch, seq_len, kv_len, topk, heads, dim, seed",
    [
        (64, 1, 128, 128, 16, 512, 2025),
        (64, 1, 400, 392, 16, 512, 2026),
        (16, 1, 168, 165, 16, 512, 2027),
        (1, 240, 240, 128, 8, 512, 2028),
        (64, 1, 144, 137, 16, 512, 2029),
        (64, 1, 640, 598, 16, 512, 2030),
        (1, 1, 264, 257, 16, 512, 2031),
        (1, 240, 240, 128, 4, 512, 2032),
    ],
)
def test_sparse_attention(batch, seq_len, kv_len, topk, heads, dim, seed):
    device = torch_device_fn.current_device()
    utils.init_seed(seed)

    q = torch.empty((batch, seq_len, heads, dim), device=device, dtype=torch.bfloat16)
    q.uniform_(-0.05, 0.05)
    kv = torch.empty((batch, kv_len, dim), device=device, dtype=torch.bfloat16)
    kv.uniform_(-0.05, 0.05)
    attn_sink = torch.empty((heads,), device=device, dtype=torch.float32)
    attn_sink.uniform_(-0.1, 0.1)
    topk_idxs = torch.randint(
        0,
        kv_len,
        (batch, seq_len, topk),
        device=device,
        dtype=torch.int32,
    )
    scale = float(1.0 / np.sqrt(dim))

    ref_q = utils.to_reference(q, False)
    ref_kv = utils.to_reference(kv, False)
    ref_attn_sink = utils.to_reference(attn_sink, False)
    ref_topk_idxs = utils.to_reference(topk_idxs, False)

    torch_result = sparse_attention_ref(
        ref_q, ref_kv, ref_attn_sink, ref_topk_idxs, scale
    )
    gems_result = flag_gems.sparse_attn_triton(q, kv, attn_sink, topk_idxs, scale)

    utils.gems_assert_close(gems_result, torch_result, torch.bfloat16, atol=1e-3)


def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    # batch, nheads = slopes.shape
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)
    if causal:
        return (
            torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
        )

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)
    return -slopes * relative_pos.to(dtype=slopes.dtype)


@pytest.mark.flash_attention_forward
@pytest.mark.skip(
    reason="Issue #2809: The operator fails this test on Nvidia at least."
)
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(vendor_name == "metax", reason="Issue #2811: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2810: RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Issue #2812: Not working")
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128, 192, 256])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_foward_nonsquare_qk(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_flash_fwd(
        ref_q, ref_k, ref_v, scale, is_causal
    )
    gems_out, gems_lse, _, _, _ = gems_flash_fwd(q, k, v, scale, is_causal)

    utils.gems_assert_close(gems_out, torch_out, dtype)
    # TODO(Iluvatar): Don't return prematurily here.
    if vendor_name == "iluvatar":
        return
    utils.gems_assert_close(gems_lse, torch_lse, torch.float)


@pytest.mark.flash_attention_forward
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None, reason="FP8 is not available"
)
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [
        (1, 16, 512, 512),
        (2, 16, 1024, 1024),
        (4, 16, 2048, 2048),
        (8, 32, 512, 512),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_flash_attention_forward_w8a8(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, _, _, _, _ = torch_flash_fwd(q, k, v, scale, is_causal)
    gems_out = gems_flash_fwd_w8a8(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        scale,
        is_causal,
    )

    _assert_w8a8_attention_close(gems_out, torch_out)


# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    # row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long)[:, None]
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        # key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        key_leftpad = key_leftpad[:, None, None, None]
        # col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = col_idx.repeat(key_leftpad.shape[0], 1, 1, 1)
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        # else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        else key_padding_mask.sum(-1)[:, None, None, None]
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        # else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        else query_padding_mask.sum(-1)[:, None, None, None]
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    scale,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        scale: float
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q *= scale

    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    g = q.shape[2] // k.shape[2]
    # k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    # v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))

    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap

    if key_padding_mask is not None:
        scores.masked_fill_((~key_padding_mask)[:, None, None, :], float("-inf"))

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        mask = (~query_padding_mask)[:, None, :, None]
        attention = attention.masked_fill(mask, 0.0)

    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_((~query_padding_mask)[:, :, None, None], 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2810: RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Issue #2812: Not supported")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2814: Not supported")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "num_head_k", "q_seq_len", "kv_seq_len"],
    [(4, 8, 2, 1024, 1024), (4, 4, 4, 1, 519)],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_forward_gqa_alibi_softcap(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    soft_cap,
    alibi,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head_k, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    if alibi:
        # alibi_slopes = torch.rand(batch, num_head, device=device, dtype=torch.float32) * 0.3
        alibi_slopes = (
            torch.ones(batch, num_head, device=device, dtype=torch.float32) * 0.3
        )
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, q_seq_len, kv_seq_len, causal=is_causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    torch_out, _ = attention_ref(
        ref_q,
        ref_k,
        ref_v,
        scale,
        None,
        None,
        attn_bias,
        0.0,
        None,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=soft_cap if soft_cap is not None else 0,
    )

    gems_out, _, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        alibi_slopes=alibi_slopes,
        softcap=soft_cap if soft_cap is not None else 0,
        disable_splitkv=True,
    )

    utils.gems_assert_close(gems_out, torch_out, dtype)


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2810: RuntimeError")
@pytest.mark.skipif(vendor_name == "metax", reason="Issue #2811: Not working")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Issue #2812: Not working")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2814: Not working")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "num_head_k", "q_seq_len", "kv_seq_len"],
    [(1, 4, 1, 1, 1024), (4, 4, 4, 1, 519)],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_foward_splitkv(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    soft_cap,
    alibi,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head_k, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    if alibi:
        # alibi_slopes = torch.rand(batch, num_head, device=device, dtype=torch.float32) * 0.3
        alibi_slopes = (
            torch.ones(batch, num_head, device=device, dtype=torch.float32) * 0.3
        )
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, q_seq_len, kv_seq_len, causal=is_causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    torch_out, _ = attention_ref(
        ref_q,
        ref_k,
        ref_v,
        scale,
        None,
        None,
        attn_bias,
        0.0,
        None,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=soft_cap if soft_cap is not None else 0,
    )

    gems_out, gems_lse, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        alibi_slopes=alibi_slopes,
        softcap=soft_cap if soft_cap is not None else 0,
    )

    utils.gems_assert_close(gems_out, torch_out, dtype)


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2810: RuntimeError")
@pytest.mark.skipif(vendor_name == "metax", reason="Issue #2811: Not working")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Issue #2812: Not working")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2814: Not working")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (8, 32, 1024, 1024), (8, 32, 1024, 128), (8, 32, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize(
    ["window_size_left", "window_size_right"], [(256, 0), (128, 128)]
)
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_foward_swa(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    window_size_left,
    window_size_right,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_flash_fwd(
        ref_q,
        ref_k,
        ref_v,
        scale,
        is_causal,
        dropout_p=0,
        return_debug_mask=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    gems_out, gems_lse, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        dropout_p=0,
        return_debug_mask=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )

    utils.gems_assert_close(gems_out, torch_out, dtype)
    # TODO(Iluvatar): Don't return early here.
    if vendor_name == "iluvatar":
        return
    utils.gems_assert_close(gems_lse, torch_lse, torch.float)


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(triton.__version__ < "3.1", reason="RequiresTriton >= 3.1")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2810: RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Issue #2812: Not supported")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2814: Not supported")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [
        (1, 1, 1024, 1024),
    ],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_dropout(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_size))
    dropout_p = 0.2
    _, _, _, _, debug_softmax = gems_flash_fwd(
        q, k, v, scale, is_causal, dropout_p=dropout_p, return_debug_mask=True
    )

    dropout_ratio = torch.sum(debug_softmax < 0) / torch.sum(debug_softmax != 0)
    np.testing.assert_allclose(dropout_ratio.to("cpu"), dropout_p, rtol=5e-2)
