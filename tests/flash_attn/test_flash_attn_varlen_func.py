# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import pytest
import torch

from tests.utils import format_tc
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

NUM_HEADS = [(8, 2)]
HEAD_SIZES = [64, 128, 192, 256, 512]
BLOCK_SIZES = [64, 128]
DTYPES = [torch.bfloat16]
QDTYPES = [None]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [2048]
SOFT_CAPS = [None]
SLIDING_WINDOWS = [(-1, 127), (127, -1), (64, 64), (-1, -1)]
SINK = [False, True]
CASUAL = [False, True]
PAGED = [False, True]
FP8KV = [torch.float8_e4m3fn, None]


def ref_paged_attn(query: torch.Tensor,
                   key_cache: torch.Tensor,
                   value_cache: torch.Tensor,
                   query_lens: list[int],
                   kv_lens: list[int],
                   block_tables: torch.Tensor,
                   scale: float,
                   window_size_left: Optional[int] = None,
                   window_size_right: Optional[int] = None,
                   soft_cap: Optional[float] = None,
                   is_paged: Optional[bool] = True,
                   casual: Optional[bool] = False,
                   sink: Optional[torch.Tensor] = None,
                   q_descale: Optional[torch.Tensor] = None,
                   k_descale: Optional[torch.Tensor] = None,
                   v_descale: Optional[torch.Tensor] = None,
                   is_fp8kv: bool = False,
                   is_fp8_query: bool = False,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    if is_paged:
        _, block_size, num_kv_heads, head_size = key_cache.shape
        v_head_size = value_cache.shape[-1]
    else:
        _, num_kv_heads, head_size = key_cache.shape
        v_head_size = value_cache.shape[-1]

    if is_fp8_query:
        query = (query.to(torch.float32) * q_descale).to(dtype)

    outputs: list[torch.Tensor] = []
    start_idx = 0
    start_idx_kv = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len] * scale

        if is_paged:
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            block_indices = block_tables[i, :num_kv_blocks]

            k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
            k = k[:kv_len]
            v = value_cache[block_indices].view(-1, num_kv_heads, v_head_size)
            v = v[:kv_len]
        else:
            k = key_cache[start_idx_kv:start_idx_kv + kv_len]
            v = value_cache[start_idx_kv:start_idx_kv + kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1],
                                        dim=1).contiguous()
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1],
                                        dim=1).contiguous()

        if is_fp8kv:
            k = (k.to(torch.float32) * k_descale).to(dtype)
            v = (v.to(torch.float32) * v_descale).to(dtype)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if window_size_right > 0 or window_size_left > 0:
            if window_size_right < 0:
                window_size_right = max(kv_lens)
            if window_size_left < 0:
                window_size_left = max(kv_lens)

            mask_right = torch.triu(empty_mask,
                                    diagonal=kv_len - query_len +
                                    window_size_right + 1).bool()
            mask_left = torch.triu(empty_mask,
                                   diagonal=kv_len - query_len -
                                   window_size_left).bool().logical_not()
            mask_local = mask_right | mask_left
            attn.masked_fill_(mask_local, float("-inf"))
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if casual:
            attn.masked_fill_(mask, float("-inf"))
        if sink is not None:
            sink_expanded = sink.view(sink.size()[0], 1,
                                      1).expand(attn.size()[0],
                                                attn.size()[1], 1)
            attn = torch.cat([attn, sink_expanded], dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sink is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len
        start_idx_kv += kv_len

    return torch.cat(outputs, dim=0)


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "test_varlen_with_paged_kv": {
        "seq_lens": [[(1, 1328), (5, 18), (129, 463)]],
        "head_size": [64, 128],
        "num_heads": [(8, 2)],
        "num_blocks": [64],
        "window_size": [(-1, -1), (127, 127)],
        "is_paged": [True],
        "stride_pad": [0, 32]
    },
    "test_decode_with_paged_kv": {
        "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
        "num_heads": [(8, 2)],
        "head_size": [64, 128],
        "num_blocks": [64],
        "window_size": [(-1, -1), (127, -1)],
    },
    "test_decode_with_paged_kv_mla": {
        "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
        "num_heads": [(8, 1)],
        "head_size_kv": [(192, 128)],
        "num_blocks": [2048],
    },
    "test_paged_decode_noncontiguous_qkv": {
        "seq_lens": [[(1, 1025), (1, 523), (1, 37)]],
        "num_heads": [(8, 2)],
        "head_size": [64, 128],
        "num_blocks": [2048],
        "noncontig_mode": ["interleaved_kv", "padded_head", "fused_qkv",
                           "hybrid_blocks"],
    },
}


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("window_size", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2])
@pytest.mark.parametrize("q_dtype", QDTYPES, ids=format_tc)
@pytest.mark.parametrize("is_sink", SINK)
@pytest.mark.parametrize("is_casual", CASUAL)
@pytest.mark.parametrize("is_paged", PAGED)
@pytest.mark.parametrize("fp8_dtype", FP8KV, ids=format_tc)
@pytest.mark.parametrize("stride_pad", [0, 32])
@torch.inference_mode()
def test_varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    window_size: tuple[int, int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
    q_dtype: Optional[torch.dtype],
    is_sink: bool,
    is_casual: bool,
    is_paged: bool,
    fp8_dtype: Optional[torch.dtype],
    stride_pad: int,
) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    # # FIXME: remove skip
    if (is_casual and seq_lens[1][0]
            == 5) and (os.getenv("SKIP_HANG_KERNEL") is not None
                       and os.getenv("SKIP_HANG_KERNEL") == "1"):
        pytest.skip("skip casual for seqlen0 to avoid runtime hang on CI.")
    if (window_size[0] != -1 or window_size[1]
            != -1) and (os.getenv("SKIP_HANG_KERNEL") is not None
                        and os.getenv("SKIP_HANG_KERNEL") == "1"):
        pytest.skip("skip local attn to avoid runtime hang on CI.")
    if block_size == 128 and num_blocks == 32768 and head_size >= 192:
        pytest.skip("skip test cases that may run out of Memory.")
    if stride_pad > 0 and fp8_dtype is not None:
        pytest.skip("non-contiguous Q/K/V with FP8 KV cache not tested")
    if stride_pad > 0 and q_dtype is not None:
        pytest.skip("non-contiguous Q/K/V with quantized query not tested")
    # if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
    #     pytest.skip("Flash attention with quantized inputs is only "
    #                 "supported on version 3 with bfloat16 base type")
    torch.manual_seed(4242)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    if stride_pad > 0:
        padded_head = head_size + stride_pad
        query_padded = torch.randn(sum(query_lens),
                                   num_query_heads,
                                   padded_head,
                                   dtype=dtype)
        query_padded[:, :, :head_size] = query
        query = query_padded[:, :, :head_size]
        assert not query.is_contiguous()
        assert query.stride(-1) == 1
    if is_paged:
        key_cache = torch.randn(num_blocks,
                                block_size,
                                num_kv_heads,
                                head_size,
                                dtype=dtype)
    else:
        key_cache = torch.randn(sum(kv_lens),
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    if stride_pad > 0 and not is_paged:
        padded_head = head_size + stride_pad
        k_padded = torch.randn(*key_cache.shape[:-1],
                                padded_head,
                                dtype=dtype)
        k_padded[..., :head_size] = key_cache
        key_cache = k_padded[..., :head_size]
        v_padded = torch.randn(*value_cache.shape[:-1],
                                padded_head,
                                dtype=dtype)
        v_padded[..., :head_size] = value_cache
        value_cache = v_padded[..., :head_size]
        assert not key_cache.is_contiguous()
        assert not value_cache.is_contiguous()
        assert key_cache.stride(-1) == 1
        assert value_cache.stride(-1) == 1

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    cu_kv_lens = torch.tensor([0] + kv_lens,
                              dtype=torch.int32).cumsum(dim=0,
                                                        dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    sink = None
    if is_sink:
        sink = torch.randn(num_query_heads, dtype=dtype)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None  #noqa: F841
    k_descale = None  #noqa: F841
    v_descale = None  #noqa: F841
    scale_shape = (num_seqs, num_kv_heads)
    is_fp8_query = q_dtype is not None
    if is_fp8_query:
        q_descale = (torch.abs(query).max() / 200).to(torch.float32)
        maybe_quantized_query = (query / q_descale).to(q_dtype)
    is_fp8kv = fp8_dtype is not None
    if is_fp8kv:
        k_descale = (torch.abs(key_cache).max() / 200).to(torch.float32)
        v_descale = (torch.abs(value_cache).max() / 200).to(torch.float32)
        maybe_quantized_key_cache = (key_cache / k_descale).to(fp8_dtype)
        maybe_quantized_value_cache = (value_cache / v_descale).to(fp8_dtype)

    if is_paged:
        output = flash_attn_varlen_func(maybe_quantized_query,
                                        maybe_quantized_key_cache,
                                        maybe_quantized_value_cache,
                                        max_query_len,
                                        cu_query_lens,
                                        max_kv_len,
                                        seqused_k=seq_k,
                                        q_descale=q_descale.expand(scale_shape)
                                        if q_descale is not None else None,
                                        k_descale=k_descale.expand(scale_shape)
                                        if k_descale is not None else None,
                                        v_descale=v_descale.expand(scale_shape)
                                        if v_descale is not None else None,
                                        softmax_scale=scale,
                                        causal=is_casual,
                                        block_table=block_tables,
                                        window_size=window_size,
                                        s_aux=sink)
    else:
        output = flash_attn_varlen_func(maybe_quantized_query,
                                        maybe_quantized_key_cache,
                                        maybe_quantized_value_cache,
                                        max_query_len,
                                        cu_query_lens,
                                        max_kv_len,
                                        cu_seqlens_k=cu_kv_lens,
                                        q_descale=q_descale.expand(scale_shape)
                                        if q_descale is not None else None,
                                        k_descale=k_descale.expand(scale_shape)
                                        if k_descale is not None else None,
                                        v_descale=v_descale.expand(scale_shape)
                                        if v_descale is not None else None,
                                        softmax_scale=scale,
                                        causal=is_casual,
                                        block_table=None,
                                        window_size=window_size,
                                        s_aux=sink)

    ref_output = ref_paged_attn(
        query=query.contiguous(),
        key_cache=maybe_quantized_key_cache.contiguous(),
        value_cache=maybe_quantized_value_cache.contiguous(),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        casual=is_casual,
        is_paged=is_paged,
        sink=sink,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        is_fp8kv=is_fp8kv,
        is_fp8_query=is_fp8_query,
        dtype=dtype)
    atol, rtol = 2e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    if window_size[0] != -1 or window_size[1] != -1:
        atol, rtol = 1.5e-2, 1.5e-2
    if fp8_dtype is not None:
        atol, rtol = 1.5e-2, 1.5e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"
    torch.xpu.empty_cache()


@pytest.mark.parametrize("seq_lens",
                         [[(1, 1025)], [(1, 523), (1, 37),
                                        (1, 2011)], [(1, 13000)],
                          [(1, 523), (1, 37), (1, 2011), (1, 5000)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2])
@pytest.mark.parametrize("q_dtype", QDTYPES, ids=format_tc)
@pytest.mark.parametrize("is_sink", SINK)
@pytest.mark.parametrize("fp8_dtype", FP8KV, ids=format_tc)
@pytest.mark.parametrize("window_size", SLIDING_WINDOWS)
@torch.inference_mode()
def test_decode_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
    q_dtype: Optional[torch.dtype],
    is_sink: bool,
    fp8_dtype: Optional[torch.dtype],
    window_size: tuple[int, int],
) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    # # FIXME: remove skip
    # if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
    #     pytest.skip("Flash attention with quantized inputs is only "
    #                 "supported on version 3 with bfloat16 base type")
    if head_size == 512 and block_size == 128:
        pytest.skip("skip test cases that may run out of SLM.")
    if num_heads == (16, 1) and head_size == 256:
        pytest.skip("skip test cases that may run out of SLM.")
    if block_size == 128 and num_blocks == 32768 and head_size >= 192:
        pytest.skip("skip test cases that may run out of Memory.")
    if is_sink and window_size != (-1, -1):
        pytest.skip("sink not supported with sliding window")
    torch.manual_seed(42)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)

    seq_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    sink = None
    if is_sink:
        sink = torch.randn(num_query_heads, dtype=dtype)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None  #noqa: F841
    k_descale = None  #noqa: F841
    v_descale = None  #noqa: F841
    scale_shape = (num_seqs, num_kv_heads)
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
        k_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
        v_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
    is_fp8kv = False
    if fp8_dtype is not None:
        is_fp8kv = True
        k_descale = (torch.abs(key_cache).max() / 200).to(torch.float32)
        v_descale = (torch.abs(value_cache).max() / 200).to(torch.float32)
        maybe_quantized_key_cache = (key_cache / k_descale).to(fp8_dtype)
        maybe_quantized_value_cache = (value_cache / v_descale).to(fp8_dtype)

    output = flash_attn_varlen_func(maybe_quantized_query,
                                    maybe_quantized_key_cache,
                                    maybe_quantized_value_cache,
                                    max_query_len,
                                    cu_query_lens,
                                    max_kv_len,
                                    seqused_k=seq_k,
                                    softmax_scale=scale,
                                    causal=False,
                                    block_table=block_tables,
                                    k_descale=k_descale.expand(scale_shape)
                                    if k_descale is not None else None,
                                    v_descale=v_descale.expand(scale_shape)
                                    if v_descale is not None else None,
                                    window_size=window_size,
                                    s_aux=sink)

    ref_output = ref_paged_attn(query=query,
                                key_cache=maybe_quantized_key_cache,
                                value_cache=maybe_quantized_value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=False,
                                is_paged=True,
                                sink=sink,
                                k_descale=k_descale,
                                v_descale=v_descale,
                                window_size_left=window_size[0],
                                window_size_right=window_size[1],
                                is_fp8kv=is_fp8kv,
                                dtype=dtype)
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    if fp8_dtype is not None:
        atol, rtol = 1.5e-2, 1.5e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"
    torch.xpu.empty_cache()



@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 1025)], [(1, 523), (1, 37),
                   (1, 2011)], [(1, 523), (1, 37), (1, 2011), (1, 5000)]])
@pytest.mark.parametrize("num_heads", [(8, 1), (16, 1), (8, 2)])
@pytest.mark.parametrize("head_size_kv", [(192, 128)])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_blocks", [2048])
@torch.inference_mode()
def test_decode_with_paged_kv_mla(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size_kv: tuple[int, int],
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    """Test paged decode with MLA-like KV cache layout.

    MLA stores K and V in a shared buffer of shape [..., k_head_size].
    K uses the full buffer (head_dim=192, rope + nope), while V is a
    slice of the first v_head_size dims (head_dim=128).  V is therefore
    non-contiguous with stride(-1)==1.

    The kernel computes attention scores using K (head_dim=k_head_size) and
    multiplies by V (head_dim=v_head_size), producing output with
    head_dim=v_head_size.
    """
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    torch.manual_seed(42)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    k_head_size, v_head_size = head_size_kv
    scale = k_head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        k_head_size,
                        dtype=dtype)

    # MLA-like combined KV cache: buffer has k_head_size dims.
    # K uses the full buffer (head_dim=192), V is the first v_head_size
    # dims (head_dim=128) — a non-contiguous slice.
    combined_kv_cache = torch.randn(num_blocks,
                                    block_size,
                                    num_kv_heads,
                                    k_head_size,
                                    dtype=dtype)

    key_cache = combined_kv_cache
    value_cache = combined_kv_cache[..., :v_head_size]

    assert key_cache.is_contiguous(), "key_cache should be contiguous"
    assert not value_cache.is_contiguous(), \
        "value_cache should be non-contiguous"
    assert value_cache.stride(-1) == 1

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    output = flash_attn_varlen_func(query,
                                    key_cache,
                                    value_cache,
                                    max_query_len,
                                    cu_query_lens,
                                    max_kv_len,
                                    seqused_k=seq_k,
                                    softmax_scale=scale,
                                    causal=False,
                                    block_table=block_tables,
                                    window_size=(-1, -1))

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=False,
                                is_paged=True,
                                sink=None,
                                window_size_left=-1,
                                window_size_right=-1)
    atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"
    torch.xpu.empty_cache()



@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 1025), (1, 523), (1, 37)],
     [(1, 523), (1, 37), (1, 2011), (1, 5000)]])
@pytest.mark.parametrize("num_heads", [(8, 2), (8, 1)])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_blocks", [2048])
@pytest.mark.parametrize("noncontig_mode",
                         ["interleaved_kv", "padded_head", "fused_qkv",
                          "hybrid_blocks"])
@torch.inference_mode()
def test_paged_decode_noncontiguous_qkv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    num_blocks: int,
    noncontig_mode: str,
) -> None:
    """Test paged decode with non-contiguous Q, K, V tensors.

    The paged decode kernel reads actual tensor strides for K and V, so
    it should handle non-contiguous KV cache layouts correctly.

    Non-contiguous modes:
    - interleaved_kv: K and V share a combined buffer of shape
          [num_blocks, block_size, num_kv_heads, 2*head_size].
      K = buf[..., :head_size], V = buf[..., head_size:].
      Both are non-contiguous with stride(-2) = 2*head_size.
    - padded_head: KV buffer has extra padding in the last dimension.
          [num_blocks, block_size, num_kv_heads, head_size + pad].
      K = buf[..., :head_size].  V is a separate padded buffer.
    - fused_qkv: Q is sliced from a fused QKV projection of shape
          [total_tokens, (num_q_heads + 2*num_kv_heads), head_size].
      Q is non-contiguous (stride(0) != num_q_heads * head_size) and
      must be made contiguous before entering the kernel.  KV cache is
      interleaved as in the interleaved_kv mode.
    - hybrid_blocks: Simulates vLLM hybrid model (Mamba + Attention)
          where KV cache has layout (num_blocks, 2, block_size,
          num_kv_heads, head_size).  K = buf[:, 0], V = buf[:, 1].
      stride(0) = 2 * block_size * heads * head_size, so
      block_stride_elems = 2 * block_size.  This exercises the
      page_stride_tiles != tiles_per_page code path.
    """
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    torch.manual_seed(42)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    # ── Build non-contiguous Q ──────────────────────────────────────
    if noncontig_mode == "fused_qkv":
        total_heads = num_query_heads + 2 * num_kv_heads
        fused_qkv = torch.randn(sum(query_lens),
                                 total_heads,
                                 head_size,
                                 dtype=dtype)
        q_slice = fused_qkv[:, :num_query_heads, :]
        assert not q_slice.is_contiguous(), \
            "fused Q slice should be non-contiguous"
        assert q_slice.stride(-1) == 1
        # The kernel requires Q to be contiguous (CHECK_CONTIGUOUS),
        # so we call .contiguous() explicitly, mimicking the real
        # vLLM code path after fused QKV projection + split.
        query = q_slice.contiguous()
    else:
        query = torch.randn(sum(query_lens),
                             num_query_heads,
                             head_size,
                             dtype=dtype)
    assert query.is_contiguous()

    # ── Build non-contiguous K / V cache ────────────────────────────
    if noncontig_mode == "interleaved_kv" or noncontig_mode == "fused_qkv":
        # Combined buffer: K and V interleaved in the head dimension
        combined_kv = torch.randn(num_blocks,
                                  block_size,
                                  num_kv_heads,
                                  2 * head_size,
                                  dtype=dtype)
        key_cache = combined_kv[..., :head_size]
        value_cache = combined_kv[..., head_size:]
    elif noncontig_mode == "padded_head":
        # Padded buffer: extra 32 elements padding after head_size
        pad = 32
        key_buf = torch.randn(num_blocks,
                               block_size,
                               num_kv_heads,
                               head_size + pad,
                               dtype=dtype)
        val_buf = torch.randn(num_blocks,
                               block_size,
                               num_kv_heads,
                               head_size + pad,
                               dtype=dtype)
        key_cache = key_buf[..., :head_size]
        value_cache = val_buf[..., :head_size]
    elif noncontig_mode == "hybrid_blocks":
        # True hybrid layout: K and V blocks alternate in dim 1
        # Mimics vLLM _update_hybrid_attention_mamba_layout which
        # transposes (2, num_blocks, B, H, D) -> (num_blocks, 2, B, H, D)
        hybrid_kv = torch.randn(num_blocks, 2, block_size,
                                num_kv_heads, head_size, dtype=dtype)
        key_cache = hybrid_kv[:, 0, :, :, :]
        value_cache = hybrid_kv[:, 1, :, :, :]
        # Verify stride(0) = 2 * block_size * heads * head_size
        expected_stride0 = 2 * block_size * num_kv_heads * head_size
        assert key_cache.stride(0) == expected_stride0, \
            f"Expected stride(0)={expected_stride0}, got {key_cache.stride(0)}"

    # Verify non-contiguity invariants
    assert not key_cache.is_contiguous(), \
        f"key_cache should be non-contiguous in mode={noncontig_mode}"
    assert not value_cache.is_contiguous(), \
        f"value_cache should be non-contiguous in mode={noncontig_mode}"
    assert key_cache.stride(-1) == 1
    assert value_cache.stride(-1) == 1

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    seq_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    output = flash_attn_varlen_func(query,
                                    key_cache,
                                    value_cache,
                                    max_query_len,
                                    cu_query_lens,
                                    max_kv_len,
                                    seqused_k=seq_k,
                                    softmax_scale=scale,
                                    causal=False,
                                    block_table=block_tables,
                                    window_size=(-1, -1))

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=False,
                                is_paged=True,
                                sink=None,
                                window_size_left=-1,
                                window_size_right=-1)

    atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - ref_output))}"
    torch.xpu.empty_cache()

