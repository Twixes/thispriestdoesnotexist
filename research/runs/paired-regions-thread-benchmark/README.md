# Actual 1024px training thread benchmark

**Two CPU threads were the best observed choice:** about 1.24× the one-thread throughput. Four threads provided no additional benefit in this small sequential test. The result supports trying two threads for the next fresh experiment; three timed updates per configuration do not establish long-run performance.

| CPU threads | Mean full update | Median | Min–max | Peak process RSS | Preflight free memory |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.72 s | 5.94 s | 5.28–5.95 s | 5.30 GiB | 31% |
| 2 | 4.62 s | 4.87 s | 4.07–4.92 s | 5.29 GiB | 26% |
| 4 | 4.77 s | 4.86 s | 4.40–5.07 s | 5.48 GiB | 26% |

Every process started fresh from the same original FFHQ1024 source and single-pair collar-region manifest, seed 20260921, LR `1e-4`, batch one, unchanged freeze policy, and fresh preservation weight one. Interop threads stayed at one. Each configuration ran one warmup update plus three measured updates, for exactly **twelve actual updates**. No checkpoints were retained.

The timed unchanged `paired_regions.update` includes paired and fresh source/student forwards, both backward passes, finite-gradient checks, Adam update and finite-parameter checks. Source loading, strict verification, final state hashing and final preview are timed separately in each result. Periodic checkpoint I/O is excluded. CPU time per timed update averaged 6.35/6.71/8.66 seconds for 1/2/4 threads, respectively; extra threads consumed more CPU without proportional wall-time benefit.

The previous 200-step region process and its 32-image evaluation had both finished before this benchmark started. A fresh memory guard preceded each model load. Existing unrelated training processes were not stopped or modified, so scheduling and system contention limit generalization from this small experiment.

## Numerical and preservation checks

All source/W provenance checks ran first at **one CPU thread with zero tolerance**. Repeating source mapping and synthesis after switching to the requested thread count found **zero** W error, zero floating-point image error, and zero changed uint8 channels for this calibration source in all configurations. No tolerance was relaxed.

All source/frozen-state checks passed, and each student had real finite parameter changes. The complete final student-state digest was identical across configurations. The three final native PNGs were byte-identical. Loss metrics matched exactly for one versus two threads; the largest four-thread difference was `1.1920928955078125e-7`, consistent with a small scalar reduction difference. These observations cover four updates on one pair; they do not promise bitwise equality for different images, longer runs, or another environment.

Each `threads-N/` contains full results/provenance, memory preflight, per-update metrics and a native `final-calibration-original.png`. `summary.json` compares them. Source is `research/experiments/paired_regions/benchmark_threads.py`; the image/JSON-only summary script is `summarize_thread_benchmark.py`. No live trainer, input, model weights, or production configuration was changed. The four-update preview is not a quality assessment or approved output.
