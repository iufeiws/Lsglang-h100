"""CPU tests of production LK host-worker identity and raw-pointer boundary."""
import ast
import ctypes
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def methods(trace=False):
    path = ROOT / 'python/sglang/srt/layers/moe/fused_moe_triton/layer.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FusedMoE')
    ns = dict(torch=torch, logger=logging.getLogger(__name__),
              get_bool_env_var=lambda *a, **k: trace)
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_get_processes_info', '_cpu_prefill')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    return ns


@pytest.mark.parametrize('ep,tp', [(1, 4), (2, 2), (4, 1), (2, 4)])
def test_host_worker_ids_cover_ep_and_tp(ep, tp):
    fn = methods()['_get_processes_info']
    actual = []
    for rank in range(ep * tp):
        layer = SimpleNamespace(moe_ep_size=ep, moe_ep_rank=rank // tp,
                                moe_tp_size=tp, moe_tp_rank=rank % tp)
        with patch.object(torch.cuda, 'current_device', return_value=rank):
            actual.append(fn(layer))
    assert actual == [(ep * tp, rank, rank) for rank in range(ep * tp)]


def raw(ptr, dtype, count):
    return torch.frombuffer((ctypes.c_ubyte * (count * torch.empty((), dtype=dtype).element_size())).from_address(ptr), dtype=dtype)


@pytest.mark.parametrize('trace', [False, True])
@pytest.mark.parametrize('rows', [0, 1, 6, 96])
@pytest.mark.parametrize('pattern', ['mixed', 'all_remote', 'all_local'])
def test_cpu_prefill_pointer_layout_and_empty_rank(rows, pattern, trace):
    fn = methods(trace)['_cpu_prefill']
    # Column-major and sliced storage must be packed for the pointer-only ABI.
    hidden = torch.arange(rows * 8, dtype=torch.bfloat16).reshape(8, rows).t()
    ids = torch.tensor([0, -1, 3, -1, 2, -1], dtype=torch.int32).repeat(rows, 2)[:, ::2]
    if pattern == 'all_remote':
        ids.fill_(-1)
    elif pattern == 'mixed':
        ids[:, 1::2] = -1
    weights = torch.arange(rows * 12, dtype=torch.float32).reshape(rows, 12)[:, ::2] / 10
    original_ids, original_weights = ids.clone(), weights.clone()
    seen = []
    joined = []
    expected = hidden.float() * (weights * (ids >= 0)).sum(1, keepdim=True)

    def native(bs, k, ip, wp, xp, op, pool_skip):
        assert joined
        i = raw(ip, torch.int32, bs * k).reshape(bs, k)
        w = raw(wp, torch.float32, bs * k).reshape(bs, k)
        x = raw(xp, torch.bfloat16, bs * 8).reshape(bs, 8)
        out = raw(op, torch.float32, bs * 8).reshape(bs, 8)
        torch.testing.assert_close(i, ids)
        torch.testing.assert_close(w, weights)
        torch.testing.assert_close(x, hidden)
        out.copy_(x.float() * (w * (i >= 0)).sum(1, keepdim=True))
        seen.append(bs)

    layer = SimpleNamespace(lk_moe=SimpleNamespace(cpu_prefill=native), check_nan_in_output=False,
                            layer_name='model.layers.20.mlp.experts',
                            _get_processes_info=lambda: (4, 0, 0))
    with (patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(synchronize=lambda: joined.append(True))),
          patch.object(torch.cuda, 'current_device', return_value='cpu')):
        actual = fn(layer, hidden, weights, ids)
    torch.testing.assert_close(actual, expected.to(hidden.dtype))
    assert len(seen) == (rows > 0 and pattern != 'all_remote')
    assert actual.dtype == hidden.dtype
    assert torch.equal(ids, original_ids) and torch.equal(weights, original_weights)
