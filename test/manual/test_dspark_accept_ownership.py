"""No-model regression for DSPARK results escaping reusable graph storage.

Execute the production accept methods without importing GPU model dependencies.
The msgspec carrier is replaced by a dataclass; all tensor operations and the
folded/eager branch are the production code. CUDA adds a delayed reader test.
"""

import ast
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[2] / (
    "python/sglang/srt/speculative/dspark_components/dspark_verify.py"
)
FIELDS = (
    "correct_len", "bonus", "cap_trim_lens", "commit_lens", "new_seq_lens", "out_tokens"
)


def production_methods():
    tree = ast.parse(SOURCE.read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    carrier = classes["AcceptOuts"]
    carrier.bases = []
    carrier.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), carrier]
    for cls, method in (
        ("DsparkVerifyEpilogue", "read_accept"),
        ("TargetVerifyExecutor", "accept_and_finalize"),
    ):
        nodes.append(next(n for n in classes[cls].body if isinstance(n, ast.FunctionDef) and n.name == method))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    ns = {"torch": torch, "dataclass": dataclasses.dataclass}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns


def fixture(device, requests, stride):
    ns = production_methods()
    graph = SimpleNamespace()
    for index, name in enumerate(FIELDS):
        shape = (requests + 2, stride) if name == "out_tokens" else (requests + 2,)
        dtype = torch.int32 if name in ("cap_trim_lens", "commit_lens") else torch.int64
        value = torch.arange(math.prod(shape), device=device, dtype=dtype).reshape(shape)
        setattr(graph, name + "_buf", value + index * 100)
    graph.read_accept = lambda bs: ns["read_accept"](graph, bs)
    executor = SimpleNamespace(verify_epilogue=graph)
    kwargs = dict(
        folded_accept=True, bs=requests, verify_ids_2d=None, target_logits=None,
        draft_block=None, sampling_info=None, draft_input=None, layout=None,
        prefix_lens=None, draft_tokens=None,
    )
    return ns, graph, executor, kwargs


@pytest.mark.parametrize("requests,stride", [(1, 6), (2, 6), (16, 6), (3, 5)])
def test_completed_step_survives_next_graph_replay(requests, stride):
    ns, graph, executor, kwargs = fixture("cpu", requests, stride)
    result = ns["accept_and_finalize"](executor, **kwargs)
    expected = {name: getattr(result, name).clone() for name in FIELDS}
    # Legal interleaving: step N+1 writes before the copy stream reads step N.
    for name in FIELDS:
        getattr(graph, name + "_buf").fill_(-999)
    for name in FIELDS:
        torch.testing.assert_close(getattr(result, name), expected[name])


def test_graph_read_remains_a_writable_view():
    ns, graph, _, _ = fixture("cpu", 2, 6)
    view = graph.read_accept(2)
    # The in-graph build_accept_outs(out=...) still needs these write-through views.
    for name in FIELDS:
        getattr(view, name).fill_(123)
        assert torch.all(getattr(graph, name + "_buf")[:2] == 123)
        assert not torch.any(getattr(graph, name + "_buf")[2:] == 123)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA, no model")
def test_delayed_copy_stream_reads_completed_step():
    ns, graph, executor, kwargs = fixture("cuda", 2, 6)
    producer, reader = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        result = ns["accept_and_finalize"](executor, **kwargs)
    reader.wait_stream(producer)
    copied = {}
    with torch.cuda.stream(reader):
        torch.cuda._sleep(50_000_000)
        for name in FIELDS:
            src = getattr(result, name)
            copied[name] = torch.empty_like(src, device="cpu", pin_memory=True)
            copied[name].copy_(src, non_blocking=True)
    with torch.cuda.stream(producer):
        for name in FIELDS:
            getattr(graph, name + "_buf").fill_(-999)
    reader.synchronize()
    producer.synchronize()
    for index, name in enumerate(FIELDS):
        expected = torch.arange(
            12 if name == "out_tokens" else 2, dtype=copied[name].dtype
        ).reshape(copied[name].shape) + index * 100
        torch.testing.assert_close(copied[name], expected)
