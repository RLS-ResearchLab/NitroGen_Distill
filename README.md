# NitroGen-Distill

Distillation research on [NitroGen](https://huggingface.co/nvidia/NitroGen) (NVIDIA's 494M
vision→gamepad gaming agent). `ngd/` is a clean reimplementation, state-dict compatible
with the official `ng.pt` and numerically parity-tested against the official code.

## Start

```bash
./env.sh setup      # venv + deps
./env.sh weights    # ng.pt (~2 GB) -> checkpoints/
./env.sh smoke --ckpt checkpoints/ng.pt --bf16
./env.sh test       # unit tests + numerical parity vs official implementation
./env.sh bench      # teacher latency / step-sweep -> results/
./env.sh probe      # dataset label statistics    -> results/
```

## Layout

- `ngd/model` — SigLIP2 tower → VL mixer → flow-matching DiT; `load_checkpoint()` in `loading.py`
- `ngd/data` — tokenizer, dataset action schema + conversion, video frames, synthetic batches
- `ngd/distill` — strategy interface; plan in [ROADMAP.md](ngd/distill/ROADMAP.md)
- `ngd/eval` — open-loop metrics, step/latency benchmark
- `results/` — benchmark + probe outputs (committed)

## Checkpoint facts (from ng.pt itself; docs are stale)

Action = **21 buttons** (17 physical + 4 virtual right-stick directions) + 2×2 sticks = 25 dims;
horizon **18**; 1 context frame (`action_shift=3`); siglip2-large-256 + 8-layer DiT (width 1024,
interleaved self/cross-attn); 16 Euler steps; no game conditioning.
