#!/usr/bin/env python3
"""对照: 保持张量存活 (服务端语义) 时连续注册是否成功, 并打印错误码字符串。"""
import torch
def make(seed, E=12, H=512, I=256):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 256, (E, 2*I, H//2), dtype=torch.uint8),
            torch.randint(119, 123, (E, 2*I, H//32), dtype=torch.uint8),
            torch.randint(0, 256, (E, H, I//2), dtype=torch.uint8),
            torch.randint(119, 123, (E, H, I//32), dtype=torch.uint8))
rt = torch.cuda.cudart()
pass
keep = []
for i in range(4):
    ts = make(i); keep.append(ts)          # 服务端语义: 永久存活
    err = 0
    for t in ts:
        err = int(rt.cudaHostRegister(t.data_ptr(), t.numel()*t.element_size(), 0))
        if err: break
    print(f"layer {i} (keep-alive): {'OK' if not err else f'FAIL err={err}'}")
