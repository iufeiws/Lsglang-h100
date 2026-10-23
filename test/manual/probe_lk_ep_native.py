"""Exercise the installed LK wheel without loading a model or starting SGLang.

CUDA host required. Run all four GPU workers, matching the server environment:
  torchrun --standalone --nproc-per-node=4 test/manual/probe_lk_ep_native.py --ep-size 2
Add --legacy-worker-ids to A/B the old NUMA topology, or --pool-slots 2 to
include pool allocation. This probes native cpu_prefill, not CUDA graph decode.
"""
import argparse
import json
import os


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep-size', type=int, default=2)
    parser.add_argument('--legacy-worker-ids', action='store_true')
    parser.add_argument('--pool-slots', type=int, default=0)
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    if not torch.cuda.is_available():
        parser.error('This native probe requires CUDA and the server CPU/NUMA environment.')
    if world % args.ep_size:
        parser.error('LOCAL_WORLD_SIZE must be divisible by --ep-size')
    torch.cuda.set_device(rank)
    import lk_moe

    tp = world // args.ep_size
    ep_rank = rank // tp
    workers, worker_id = world, rank
    if args.legacy_worker_ids and args.ep_size > 1:
        workers, worker_id = args.ep_size, ep_rank
    h, intermediate, experts, k = 5120, 2304 // tp, 12, 6
    cfg = lk_moe.MOEConfigV2()
    fields = dict(layer_id=0, num_processes=workers, process_id=worker_id,
                  gpu_id=rank, hidden_size=h, intermediate_size=intermediate,
                  expert_num=experts, top_k=k, max_batch_size=96,
                  max_num_seqs=96, group_min_len=10, group_max_len=96,
                  groupN=1, groupK=32, stride=32, super_batch_size=0,
                  pool_slots=args.pool_slots, pool_member_layers=1,
                  pool_cache=bool(args.pool_slots), pool_decode=False,
                  prefill_gpu=False, use_gpu_prefill=False, hf_mbox=0)
    for key, value in fields.items():
        setattr(cfg, key, value)

    def report(event, **kw):
        print(json.dumps(dict(rank=rank, event=event, **kw)), flush=True)

    report('construct', lk_version=lk_moe.__version__, config=fields)
    # Small resident expert set, actual H100 deployment H/I geometry. Positive
    # E2M1 weights avoid cancellation while still giving expert-specific outputs.
    w13 = torch.full((experts, 2 * intermediate, h // 2), 0x11, dtype=torch.uint8)
    w2 = torch.full((experts, h, intermediate // 2), 0x11, dtype=torch.uint8)
    s13 = torch.empty((experts, 2 * intermediate, h // 32), dtype=torch.uint8)
    s2 = torch.empty((experts, h, intermediate // 32), dtype=torch.uint8)
    for e in range(experts):
        s13[e].fill_(120 + e % 3)
        s2[e].fill_(120 + e % 3)
    moe = lk_moe.MOE_MXFP4(cfg, w13.data_ptr(), w2.data_ptr(),
                          s13.data_ptr(), s2.data_ptr(), 0, 0)
    report('constructed')

    def run(x, ids, weights):
        x, ids, weights = x.contiguous(), ids.contiguous(), weights.contiguous()
        out = torch.full(x.shape, float('nan'), dtype=torch.float32)
        moe.cpu_prefill(x.shape[0], ids.shape[1], ids.data_ptr(),
                        weights.data_ptr(), x.data_ptr(), out.data_ptr(), False)
        assert torch.isfinite(out).all(), 'native output contains NaN/Inf or unwritten rows'
        return out

    for rows in (1, 6, 16, 96):
        x = torch.full((rows, h), 1 / 128, dtype=torch.bfloat16)
        weights = torch.full((rows, k), 1 / k, dtype=torch.float32)
        for pattern in ('local', 'mixed', 'remote'):
            ids = torch.arange(k, dtype=torch.int32).repeat(rows, 1)
            if pattern == 'mixed':
                ids[:, ep_rank % 2::2] = -1
                if rows > 1:
                    ids[-1].fill_(-1)
            elif pattern == 'remote':
                ids.fill_(-1)
            report('enter', rows=rows, pattern=pattern)
            # Deliberately bypass the Python empty-work shortcut to test the
            # wheel itself. A crash pinpoints this case and this worker identity.
            actual = run(x, ids, weights)
            expected = torch.zeros_like(actual)
            for slot in range(k):
                valid = ids[:, slot] >= 0
                if valid.any():
                    expected[valid] += run(x[valid], ids[valid, slot:slot+1],
                                           weights[valid, slot:slot+1])
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
            report('pass', rows=rows, pattern=pattern)
    torch.cuda.synchronize()
    report('complete')


if __name__ == '__main__':
    main()
