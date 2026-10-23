# Hand-extracted from NVIDIA/NeMo-Skills nemo_skills/utils.py.
# Only the helpers the rest of `_vendored/nemo_skills/` actually imports.
# Kept hand-curated (rather than auto-synced) because upstream `utils.py`
# pulls in `rich`, `nemo_skills.file_utils`, etc., none of which the
# evaluator/math chain needs.
# Apache-2.0; see top-level LICENSE.

from dataclasses import dataclass, is_dataclass
from pathlib import Path


def get_logger_name(file: str) -> str:
    """Derive a stable logger name from a source file path. Mirrors upstream."""
    if "/nemo_skills/" in file:
        return "nemo_skills" + file.split("nemo_skills")[1].replace("/", ".").replace(".py", "")
    return f"[external] {Path(file).stem}"


def nested_dataclass(*args, **kwargs):
    """Decorator that recursively instantiates nested dataclasses from dicts.

    Verbatim from upstream ``nemo_skills/utils.py``; the only edit is dropping
    the optional ``omegaconf.DictConfig`` branch (we never run inside Hydra).
    Adapted from the GeeksforGeeks pattern referenced in upstream.
    """

    def wrapper(check_class):
        dict_types = (dict,)

        check_class = dataclass(check_class, **kwargs)
        orig_init = check_class.__init__

        def __init__(self, *, _init_nested=False, **kwargs):
            if _init_nested:
                for name, value in kwargs.items():
                    ft = check_class.__annotations__.get(name, None)
                    if is_dataclass(ft) and isinstance(value, dict_types):
                        kwargs[name] = ft(**value, _init_nested=_init_nested)
            orig_init(self, **kwargs)

        check_class.__init__ = __init__
        return check_class

    return wrapper(args[0]) if args else wrapper
