"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
MOE V2 - picks the _lk_moe_C_<cpu-feature>.so matching the host CPU.
"""
import sys
import platform
import importlib.util
from pathlib import Path

ARCH_TYPES = {
    "avx512_amx": "AVX512 with AMX",
    "avx512_vnni": "AVX512 with VNNI",
    "avx512_base": "AVX512 Base",
    "avx2": "AVX2"
}

_HELPERS = (
    "get_device_mem_usage",
    "pool_stats",
    "pool_slot",
    "pool_ready_slot",
    "pool_snapshot_slot",
    "pool_set_det_split",
    "pool_wait_fence",
    "pool_wait_all_fences",
    "pool_verify_sample",
    "pool_layer_info",
    "pool_capture_pause",
    "pool_capture_resume",
    "nvme_filler_attach",
    "nvme_filler_attach_gather",
    "nvme_filler_detach",
    "nvme_filler_detach_all",
    "nvme_filler_stats",
    "nvme_gate",
)

def _detect_cpu_features():
    if platform.system() != "Linux":
        print("Warning: Only Linux is supported for dynamic CPU feature detection.")
        return "avx2"

    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read().lower()

        if 'amx_bf16' in cpuinfo or 'amx_tile' in cpuinfo or 'amx_int8' in cpuinfo:
            print("Detected CPU with AMX support")
            return "avx512_amx"
        if 'avx512_vnni' in cpuinfo and 'avx512f' in cpuinfo:
            print("Detected CPU with AVX512-VNNI support")
            return "avx512_vnni"
        if 'avx512f' in cpuinfo:
            print("Detected CPU with AVX512 support")
            return "avx512_base"
        if 'avx2' in cpuinfo:
            print("Detected CPU with AVX2 support")
            return "avx2"

        print("Detected CPU with basic instruction set support")
        return "avx2"
    except Exception as e:
        print(f"Error detecting CPU features: {e}")
        return "avx2"


def _load_library():
    module_dir = Path(__file__).parent
    arch_type = _detect_cpu_features()

    so_filename = f"_lk_moe_C_{arch_type}.so"
    so_path = module_dir / so_filename

    if not so_path.exists():
        for fallback_arch in ["avx512_base", "avx2"]:
            fallback_so_path = module_dir / f"_lk_moe_C_{fallback_arch}.so"
            if fallback_so_path.exists():
                print(f"Requested {arch_type} version not found, falling back to {fallback_arch}")
                so_path = fallback_so_path
                arch_type = fallback_arch
                break
        else:
            generic_so_files = list(module_dir.glob("_lk_moe_C.cpython-*.so"))
            if generic_so_files:
                so_path = generic_so_files[0]
                print(f"No architecture-specific files found, using generic file {so_path.name}")
            else:
                raise ImportError(f"No compatible _lk_moe_C library found in {module_dir}")

    print(f"Loading {so_path.name}")

    try:
        spec = importlib.util.spec_from_file_location("_lk_moe_C", str(so_path))
        if spec is None:
            raise ImportError(f"Could not create spec for {so_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # class_name mapping → C++ pybind11 class name
        classes = {
            "MOEConfigV2": "MOEConfigV2",
            # BF16 activation
            "MOE_BF16":      "MOE_BF16",
            "MOE_FP8":       "MOE_FP8",
            "MOE_WNA16":     "MOE_WNA16",
            "MOE_NVFP4":     "MOE_NVFP4",
            "MOE_MXFP4":     "MOE_MXFP4",
            # FP16 activation
            "MOE_FP16":       "MOE_FP16",
            "MOE_FP8_FP16":   "MOE_FP8_FP16",
            "MOE_WNA16_FP16": "MOE_WNA16_FP16",
            "MOE_NVFP4_FP16": "MOE_NVFP4_FP16",
            "MOE_MXFP4_FP16": "MOE_MXFP4_FP16",
        }

        result = {}
        for name, class_name in classes.items():
            if hasattr(module, class_name):
                result[name] = getattr(module, class_name)
            else:
                print(f"Warning: {class_name} not found in module")

        if "MOEConfigV2" not in result:
            raise AttributeError("MOEConfigV2 not found in the module")

        for helper in _HELPERS:
            if hasattr(module, helper):
                result[helper] = getattr(module, helper)

        print(f"Successfully loaded: {list(result.keys())}")
        return result

    except Exception as e:
        print(f"Error loading library: {e}")
        raise ImportError(f"Failed to load {so_path}")


# 导出
def get_moe_classes():
    return _load_library()