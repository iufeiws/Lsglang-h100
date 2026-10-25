#!/usr/bin/env python3
"""复现: 同一进程内连续注册多份主机专家张量时的 cudaHostRegister 失败。
用法: python repro_hostregister.py [层数] [--free-first]
"""
import sys, gc, torch

def make(seed, E=12, H=512, I=256):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8),
            torch.randint(119, 123, (E, 2 * I, H // 32), dtype=torch.uint8),
            torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8),
            torch.randint(119, 123, (E, H, I // 32), dtype=torch.uint8))

def register(ts):
    for t in ts:
        err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        if int(err) != 0:
            return t.numel() * t.element_size(), int(err)
    return None, 0

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for i in range(n):
        ts = make(i)
        kept = False
        size, err = register(ts)
        if err:
            print(f"layer {i}: FAIL size={size} err={err} (torch 解析: {torch.cuda.cudart().cudaGetErrorName(err) if hasattr(torch.cuda.cudart(),'cudaGetErrorName') else '?'})")
        else:
            print(f"layer {i}: OK  (w13 {ts[0].numel()*1} B)")
        del ts; gc.collect()

if __name__ == "__main__":
    main()
