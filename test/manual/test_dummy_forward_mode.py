"""Execute the complete production dummy builder with CPU buffers.

Only device/collective/model boundaries are stubs. These tests validate the
batch reaching attention and MoE, not CUDA kernels or LK's native threads.
"""
import ast
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from test_lk_eager_routing import Mode, SRT, scope


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Algorithm:
    def __init__(self, family):
        self.family = family

    def is_speculative(self):
        return self.family != "none"

    def is_eagle(self):
        return self.family == "eagle"

    def is_standalone(self):
        return False

    def is_dflash_family(self):
        return self.family in ("dspark", "dflash")

    def is_ngram(self):
        return False

    def supports_target_verify_for_draft(self):
        return self.is_dflash_family()


Hidden = NS(NULL=0, FULL=1)


def build_dummy(*, family="dspark", draft=False, override=None, width=6,
                batch_size=2, pd_prefill=False, extend_width=None,
                gather=False, raise_in_model=False, capacity=None):
    # Deliberately keep configured target width at 6 even when draft width is 5.
    spec_config = NS(speculative_num_draft_tokens=6, speculative_num_steps=5,
                     speculative_eagle_topk=1)
    create_verify = load_function(
        SRT / "speculative/spec_info.py", "create_dummy_verify_input",
        {"get_spec_config": lambda: spec_config},
    )
    seen = NS(metadata=[], model=[], verify_calls=[], dp=[])
    algorithm = Algorithm(family)

    def verify(*args):
        seen.verify_calls.append(args[2])
        return create_verify(*args)

    def model(input_ids, positions, batch, **kwargs):
        assert scope.get_forward_batch() is batch
        seen.model.append(batch)
        seen.moe_decode = scope.is_decode_or_verify_batch()
        seen.moe_width = scope.get_verify_rows_per_request(input_ids.numel())
        if raise_in_model:
            raise ValueError("model failure")

    mr = NS(is_generation=True, is_draft_worker=draft, spec_algorithm=algorithm,
            decode_num_tokens_per_req=lambda: width, model=NS(forward=model),
            model_config=NS(hidden_size=8), device="cpu", dtype=torch.bfloat16,
            attn_backend=NS(get_cuda_graph_seq_len_fill_value=lambda: 1,
                            init_forward_metadata=seen.metadata.append),
            kv_index_translator=NS(bind_runner_slots=lambda batch: None),
            lora_manager=None, prepare_dummy_forward_batch=lambda batch: batch)
    rows = capacity if capacity is not None else batch_size * max(width, extend_width or 1)
    buffers = NS(input_ids=torch.zeros(rows, dtype=torch.int64),
                 positions=torch.zeros(rows, dtype=torch.int64),
                 out_cache_loc=torch.zeros(rows, dtype=torch.int64),
                 mrope_positions=torch.zeros(3, rows, dtype=torch.int64),
                 num_token_non_padded=torch.zeros((), dtype=torch.int32),
                 req_pool_indices=torch.zeros(batch_size, dtype=torch.int64),
                 seq_lens=torch.ones(batch_size, dtype=torch.int32),
                 seq_lens_cpu=torch.ones(batch_size, dtype=torch.int32),
                 next_token_logits_buffer=None, encoder_lens=None,
                 global_num_tokens_gpu=torch.zeros(2 if gather else 1, dtype=torch.int32),
                 global_num_tokens_for_logprob_gpu=torch.zeros(2 if gather else 1, dtype=torch.int32),
                 custom_mask=torch.ones(rows * 16, dtype=torch.bool),
                 ngram_embedding_info=None)
    namespace = dict(torch=torch, ForwardMode=Mode, CaptureHiddenMode=Hidden,
                     get_server_return_hidden_states_mode=lambda: Hidden.NULL,
                     get_disagg=lambda: NS(disaggregation_mode="prefill" if pd_prefill else "null"),
                     get_flags=lambda: NS(capture=NS(enable_torch_compile=False)),
                     get_parallel=lambda: NS(pp_size=1, num_dp_ranks=2,
                                             tp_group=NS(barrier=lambda: None)),
                     require_mlp_tp_gather=lambda: gather,
                     require_attn_tp_gather=lambda: False,
                     require_gathered_buffer=lambda: gather,
                     create_dummy_verify_input=verify, ForwardBatch=NS,
                     DpPaddingMode=NS(get_default_mode_in_cuda_graph=lambda: NS(is_max_len=lambda: True)),
                     enable_num_token_non_padded=lambda: True,
                     get_exec=lambda: NS(features=NS(enable_encoder_swa_bounded_replay=False)),
                     set_dp_buffer_len=lambda *args: seen.dp.append(args),
                     set_is_extend_in_batch=lambda value: None,
                     forward_context=lambda ctx: nullcontext(), ForwardContext=NS,
                     set_forward_batch=scope.set_forward_batch, empty_context=nullcontext)
    dummy = load_function(SRT / "model_executor/runner/base_runner.py", "_dummy_run",
                          namespace, "BaseRunner")
    modules = {}
    for module_name, cls_name in [("model_executor.forward_batch_info", "CaptureHiddenMode"),
                                  ("speculative.dflash_info", "DFlashVerifyInput"),
                                  ("speculative.eagle_info", "EagleVerifyInput")]:
        module = ModuleType("sglang.srt." + module_name)
        setattr(module, cls_name, Hidden if cls_name == "CaptureHiddenMode" else NS)
        modules[module.__name__] = module
    outer = NS(forward_mode=Mode.EXTEND, spec_info=None)
    with patch.dict("sys.modules", modules), scope.set_forward_batch(outer):
        try:
            dummy(NS(model_runner=mr), batch_size, buffers=buffers,
                  forward_mode_override=override,
                  extend_num_tokens_per_req=extend_width)
        finally:
            assert scope.get_forward_batch() is outer
    return seen


@pytest.mark.parametrize("family", ["none", "dspark", "dflash", "eagle"])
@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("batch_size", [1, 16])
def test_explicit_decode_is_one_row_without_verify_metadata(family, gather, batch_size):
    seen = build_dummy(family=family, override=Mode.DECODE, gather=gather,
                       batch_size=batch_size, capacity=batch_size)
    batch = seen.model[0]
    assert seen.metadata == [batch]
    assert batch.forward_mode == batch.global_forward_mode == Mode.DECODE
    assert batch.input_ids.numel() == batch_size
    assert batch.num_token_non_padded.item() == batch_size
    assert batch.spec_info is None and seen.verify_calls == []
    assert seen.moe_decode and seen.moe_width == 1
    if gather:
        assert batch.global_num_tokens_cpu == [batch_size, batch_size]
        assert batch.global_num_tokens_gpu.tolist() == [batch_size, batch_size]


@pytest.mark.parametrize("family,draft,width", [("dspark", False, 6), ("dspark", True, 5),
                                               ("dspark", True, 6), ("dflash", True, 6),
                                               ("eagle", False, 6)])
@pytest.mark.parametrize("explicit", [False, True])
def test_verify_rows_and_metadata_share_runner_width(family, draft, width, explicit):
    seen = build_dummy(family=family, draft=draft, width=width,
                       override=Mode.TARGET_VERIFY if explicit else None)
    batch = seen.model[0]
    assert seen.metadata == [batch]
    assert batch.forward_mode == batch.global_forward_mode == Mode.TARGET_VERIFY
    assert batch.input_ids.numel() == batch.batch_size * width
    assert batch.num_token_non_padded.item() == batch.batch_size * width
    assert batch.spec_info.draft_token_num == width
    assert seen.verify_calls == [width]
    assert seen.moe_decode and seen.moe_width == width
    if family == "eagle":
        assert batch.spec_info.hidden_states.shape == (2 * width, 8)


def test_dspark_draft_decode_does_not_create_block_verify_metadata():
    seen = build_dummy(draft=True, override=Mode.DECODE, width=5, capacity=2)
    assert seen.model[0].spec_info is None
    assert seen.moe_width == 1


@pytest.mark.parametrize("pd_prefill", [False, True])
def test_plain_decode_default(pd_prefill):
    seen = build_dummy(family="dspark" if pd_prefill else "none", pd_prefill=pd_prefill)
    batch = seen.model[0]
    assert batch.forward_mode == Mode.DECODE
    assert batch.input_ids.numel() == 2 and batch.spec_info is None


@pytest.mark.parametrize("pd_prefill", [False, True])
def test_explicit_extend_has_no_verify_metadata(pd_prefill):
    seen = build_dummy(family="dspark" if pd_prefill else "none", pd_prefill=pd_prefill,
                       override=Mode.EXTEND, extend_width=3)
    batch = seen.model[0]
    assert batch.forward_mode == Mode.EXTEND
    assert batch.input_ids.numel() == batch.extend_num_tokens == 6
    assert batch.extend_seq_lens_cpu == [3, 3]
    assert batch.extend_start_loc.tolist() == [0, 3]
    assert batch.spec_info is None and not seen.moe_decode


def test_model_failure_restores_outer_batch():
    with pytest.raises(ValueError, match="model failure"):
        build_dummy(override=Mode.DECODE, raise_in_model=True)


def test_default_verify_rejects_buffer_too_small():
    with pytest.raises(AssertionError, match="needs a static buffer"):
        build_dummy(capacity=2)
