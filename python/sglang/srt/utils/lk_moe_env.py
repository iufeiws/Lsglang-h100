"""lk_moe configuration surface: env parsing, layer predicates and the
expert-pool pump park helper.

Everything LK_* lives here so the knobs have a single source of truth; callers
keep importing the names from ``sglang.srt.utils.common`` (compat re-export).
"""

from contextlib import contextmanager
from dataclasses import dataclass

from sglang.srt.utils.env_vars import (
    get_bool_env_var,
    get_int_env_var,
    get_str_env_var,
)


@dataclass(frozen=True)
class LkMoeConfig:
    """Immutable snapshot of the LK_* knobs for one call site.

    Built on demand (no global cache): several call sites mutate the
    environment at runtime (tests, migrate_lk_env_vars, set_int_env_var), so a
    cached singleton would go stale.
    """

    enabled: bool
    numa_interleave: bool
    pool_slots: int
    pool_enabled: bool
    pool_decode_enabled: bool
    gpu_prefill_min_tokens: int
    gpu_prefill_engine: str
    prefill_debug: bool
    layerwise_load: bool


def get_lk_moe_config() -> LkMoeConfig:
    return LkMoeConfig(
        enabled=is_lk_moe_feature_enabled(),
        numa_interleave=is_numa_interleave_enabled(),
        pool_slots=get_pool_slots(),
        pool_enabled=is_lk_pool_enabled(),
        pool_decode_enabled=is_lk_pool_decode_enabled(),
        gpu_prefill_min_tokens=get_gpu_prefill_min_tokens(),
        gpu_prefill_engine=get_gpu_prefill_engine(),
        prefill_debug=is_lk_prefill_debug(),
        layerwise_load=enabled_layerwise_load(),
    )



@contextmanager
def park_lk_moe_expert_pool():
    """Park lk_moe's expert-pool pump around CUDA-graph capture/replay.

    The pump issues cudaMemcpyAsync from its own thread; overlapping that with
    cuGraphLaunch (or with an in-flight capture) deadlocks inside libcuda on
    P2P-patched drivers.  No-op when lk_moe is off or too old to expose the
    pause/resume hooks.
    """
    if not is_lk_pool_enabled():
        yield
        return
    import lk_moe

    # Pool is on, so these hooks are part of the lk_moe contract: a build that
    # lacks them is incompatible and must fail loudly here instead of silently
    # skipping the park (which would bring the cuGraphLaunch/copy deadlock
    # back).
    pause = lk_moe.pool_capture_pause
    resume = lk_moe.pool_capture_resume
    pause()
    try:
        yield
    finally:
        resume()


def wait_lk_moe_pool_fences(stream_ptr: int) -> None:
    """Attach the expert-pool copy fences to ``stream_ptr`` (pre-replay).

    With the deterministic split on, ``publish_snapshot`` publishes CLAIMED
    slots as soon as the pump's batch fence event is recorded, while the
    copies may still be in flight on the pump stream.  A graph replay that
    reads those slots must therefore stream-wait on the recorded fence
    first; ``pool_wait_all_fences`` does exactly nothing unless det_split
    is active.  Old lk_moe builds without the hook simply keep the
    pre-feature behavior.
    """
    if not is_lk_pool_enabled():
        return
    import lk_moe

    wait = getattr(lk_moe, "pool_wait_all_fences", None)
    if wait is not None:
        wait(stream_ptr)


def is_lk_moe_feature_enabled() -> bool:
    return get_bool_env_var("LK_MOE_HYBRID_ENABLED")

def is_numa_interleave_enabled() -> bool:
    # Doctrine: tier power follows the master switch - default on; 0 opts out.
    return get_bool_env_var("LK_NUMA_INTERLEAVE", default="true")

def get_gpu_prefill_min_tokens() -> int:
    # Dual knob (designs/lk_switch_doctrine.md rule 2): default 0 = tier off,
    # pure CPU prefill; >0 both enables the tier and sets the batch gate.
    # Forwarded to lk_moe via cfg; lk_moe itself never reads the env.
    return get_int_env_var("LK_GPU_PREFILL_MIN_TOKENS", 0)

def get_gpu_prefill_engine() -> str:
    # Pure selector switch (lk_switch_doctrine: a selector never flips the
    # tier itself -- the GPU-prefill tier still powers on only via
    # LK_GPU_PREFILL_MIN_TOKENS>0 under the master switch). Chooses WHICH
    # GPU-prefill implementation a powered tier uses:
    #   "triton" (default)  = staging group-GEMM over pool slots (legacy,
    #                         fp32 activations, matches the CPU reference)
    #   "cutlass"           = restored CUTLASS grouped-GEMM engine (bf16
    #                         intermediates; NOT default: its extra bf16
    #                         rounding drifts long-context logits and flips
    #                         borderline EOS -- see journal 2026-10-22)
    # Inert unless the tier is on. Forwarded to lk_moe via cfg.use_gpu_prefill.
    val = get_str_env_var("LK_GPU_PREFILL_ENGINE", "triton").strip().lower()
    return "cutlass" if val == "cutlass" else "triton"

def get_pool_window() -> int:
    # Tuned default (hit ~81%). The ONLY pool switch is LK_POOL_SLOTS: any
    # illegal value (incl. <=0) clamps to the minimum 8, never disables --
    # a perf knob must not secretly flip the membership switch.
    return max(8, get_int_env_var("LK_POOL_WINDOW", 32))

def is_lk_gpu_prefill_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_gpu_prefill_min_tokens() > 0

def get_super_chunk_size() -> int:
    if not is_lk_gpu_prefill_enabled():
        return 0
    return get_int_env_var("LK_GPU_PREFILL_SUB_M")

def get_pool_inflight() -> int:
    # In-flight DMA jobs / event-ring depth of the decode pump; the
    # backpressure cap is 8x this -- 12 false-positives at small batch.
    return get_int_env_var("LK_POOL_INFLIGHT", 16)

def get_verify_resident() -> int:
    # VR: draft-row pool-resident experts computed GPU-side; has an acceptance
    # fidelity tradeoff -> python-owned cfg value (LK_* env is the script UI).
    return get_int_env_var("LK_VERIFY_RESIDENT", 0)


def get_verify_skip_miss() -> int:
    # verify rows drop CPU-side misses (distribution declaration, not bitwise).
    return get_int_env_var("LK_VERIFY_SKIP_MISS", 0)


def get_hf_mbox() -> int:
    # mailbox early-start decode path (default on; 0 = pure host-node path).
    return get_int_env_var("LK_HF_MBOX", 1)


def get_pool_log() -> int:
    # dstep interval of the C++ [pool] stats line (0 = silent).
    return get_int_env_var("LK_POOL_LOG")

def is_lk_prefill_debug() -> bool:
    # Extra tracing inside the prefill staging tier.
    return get_bool_env_var("LK_PP_DEBUG")

def get_gpu_prefetch_window() -> int:
    # Triton staging-ring depth (NOT a cross-layer prefetch: staging always
    # starts after this layer's topk). 1 (default) = one block, copy/gemm
    # serialize; 2 = two blocks, the next stage's H2D can overlap the previous
    # layer's gemm - only worth it with fast PCIe. Ignored by cutlass.
    return get_int_env_var("LK_GPU_PREFETCH_WINDOW", 1)


def get_prefill_wave_experts() -> int:
    # CUTLASS wave width: -1 (default) = auto -- the largest W whose deq
    # blocks plus whole tail staging layers still fit the borrow slab,
    # clamped to E (adaptive for E=256/384/512); 0 = single-shot gemm; >0 =
    # explicit cap. Waves run inside each SUB_M pass. Ignored by triton.
    return get_int_env_var("LK_GPU_PREFILL_WAVE_EXPERTS", -1)


def get_model_type_from_layer_name(layer_name: str) -> str:
    if not layer_name:
        return "main"
    
    if layer_name.startswith('stages.'):
        return "dspark"
    if layer_name.startswith('mtp.') or '.mtp.' in layer_name:
        return "mtp"
    return "main"

def get_gpu_resident_env_var(model_type: str = "main") -> str | None:
    if model_type in ("dspark", "mtp"):
        env_value = get_str_env_var("LK_GPU_RESIDENT_MOE_LAYERS_SPEC", None)
        if env_value is not None:
            return env_value

        role_env = (
            "LK_GPU_RESIDENT_MOE_LAYERS_MTP"
            if model_type == "mtp"
            else "LK_GPU_RESIDENT_MOE_LAYERS_DSPARK"
        )
        env_value = get_str_env_var(role_env, None)
        if env_value is not None:
            return env_value

    return get_str_env_var("LK_GPU_RESIDENT_MOE_LAYERS", None)

def is_lk_ple_probe_enabled() -> bool:
    """PLE ngram-row locality/traffic probe master switch."""
    return get_bool_env_var("LK_PLE_PROBE")

def get_lk_ple_probe_dump_path() -> str | None:
    """Probe row-dump path, or None (empty string means off)."""
    return get_str_env_var("LK_PLE_PROBE_DUMP", None) or None

def is_lk_moe_gpu_resident_layer(layer_id: str, model_type: str = "main") -> bool:
    if not is_lk_moe_feature_enabled():
        return True
     
    disabled_layers_env = get_gpu_resident_env_var(model_type)
    if not disabled_layers_env:
        return False   
    
    disabled_layers_env = disabled_layers_env.strip()
    
    disabled_layers = set()
    for part in disabled_layers_env.split(','):
        part = part.strip()
        if not part:
            continue
        
        if '-' in part:
            try:
                start, end = map(int, part.split('-')) 
                if start <= end:
                    disabled_layers.update(range(start, end + 1))
            except ValueError: 
                continue
        else:
            try:
                disabled_layers.add(int(part))
            except ValueError: 
                continue
     
    return layer_id in disabled_layers

def get_lk_pool_layer_set() -> set | None:
    # Same grammar as the resident-layer env; None (unset) = all non-resident
    # layers get a pool. The engine never reads this env: membership lands on
    # MOEConfigV2.gpu_pool at construction time.
    layers_env = get_str_env_var("LK_POOL_LAYERS", None)
    if not layers_env:
        return None

    pool_layers = set()
    for part in layers_env.strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start <= end:
                    pool_layers.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                pool_layers.add(int(part))
            except ValueError:
                continue
    return pool_layers


def get_pool_slots() -> int:
    return get_int_env_var("LK_POOL_SLOTS", 0)

def is_lk_pool_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_pool_slots() > 0

def is_lk_pool_layer(layer_id: str, model_type: str = "main") -> bool:
    if model_type != "main":
        return False
    if is_lk_moe_gpu_resident_layer(layer_id, model_type):
        return False
    return is_lk_pool_enabled()

def lk_route_probe_enabled() -> bool:
    return is_lk_pool_enabled() and get_bool_env_var("LK_ROUTE_PROBE")

def is_lk_pool_decode_enabled() -> bool:
    if not is_lk_moe_feature_enabled():
        return False
    return get_bool_env_var("LK_POOL_DECODE",
                            "true" if is_lk_pool_enabled() else "false")

def enabled_layerwise_load() -> bool:
    return get_bool_env_var("LK_MOE_LAYERWISE_LOAD")

