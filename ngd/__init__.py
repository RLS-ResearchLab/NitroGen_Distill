"""NitroGen-Distill: readable reimplementation of NitroGen, structured for distillation research.

Layout:
    ngd.model    -- the NitroGen policy (SigLIP vision tower + VL mixer + flow-matching DiT),
                    state-dict compatible with the official ``ng.pt`` checkpoint.
    ngd.data     -- tokenizer, action parquet schema, video frame extraction, dataset, synthetic batches.
    ngd.distill  -- distillation strategy interface + registry (implementations land here).
    ngd.eval     -- open-loop action metrics and sampling-step/latency benchmarks.
"""

__version__ = "0.1.0"
