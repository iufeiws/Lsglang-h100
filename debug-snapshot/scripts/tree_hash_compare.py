#!/usr/bin/env python3
"""Compare .py files between the repo tree and the deployed site-packages tree."""
import hashlib, os, sys

repo = "/mnt/SSD01/models/.llm-env/Lsglang-h100/python/sglang"
dep = "/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang"

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

n = 0
diff, missing = [], []
for root, dirs, files in os.walk(repo):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fn in files:
        if not fn.endswith(".py"):
            continue
        rel = os.path.relpath(os.path.join(root, fn), repo)
        rp, dp = os.path.join(repo, rel), os.path.join(dep, rel)
        n += 1
        if not os.path.exists(dp):
            missing.append(rel); continue
        a, b = md5(rp), md5(dp)
        if a != b:
            diff.append((rel, a, b))
print(f"repo .py files: {n}   differing: {len(diff)}   missing in deploy: {len(missing)}")
for rel, a, b in sorted(diff):
    print(f"DIFF {rel}  repo={a}  deploy={b}")
for rel in sorted(missing):
    print(f"MISSING {rel}")
