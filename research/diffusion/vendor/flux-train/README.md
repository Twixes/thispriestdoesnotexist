# Pinned FLUX.2 klein LoRA smoke trainer

`upstream.py` is the unchanged official Diffusers trainer at commit
`9f1246971270c84dcbe71233edb7a519596a5d02`, SHA256
`83bccc8c041496aa33939b6ff1f95d0929c1c71a9a29cdc64ecacd4a503fccfe`.
Its Apache license and source URL/hash are retained alongside it.

`run_training.py` is a separately named adaptation for exactly twenty MPS updates.
`prepare_patch.py` deterministically rebuilds it and `upstream.patch` from the unchanged
source. `patch-provenance.json` pins both the resulting trainer and `smoke_guards.py`.
Use Python 3.11 from the isolated training environment. The preparation step parses
syntax only; it does not import a trainer or load weights.

Changes preserve the upstream flow-matching objective, optimizer and FP32 LoRA parameters:

- Qwen loads initially in the existing weight dtype. The transformer already did so in
  this exact upstream version. VAE initial FP32 loading is deliberately retained because
  batch-normalization statistics are computed before its existing later dtype cast.
- Single-block LoRA targets follow the actual config (twenty blocks for base4B).
- Cached text pipeline is deleted along with its encoder/tokenizer names.
- Cached embeddings/latents, loss, gradients, clipped norm and updated parameters must
  remain finite. Missing trainable gradients or gradients on frozen parameters fail.
- Only LoRA A/B matrices may train, in FP32. Full frozen state tensors are SHA-hashed
  before/after the run; every frozen tensor must remain byte-equivalent. At least one
  adapter tensor must change. Matched module names and trainable counts are recorded.
- Saved checkpoint and final safetensor key sets must exactly match the PEFT adapter
  state prefixed by `transformer.`; saved values must be finite. Key/shape/dtype/hash
  evidence accompanies every save. This is save-content verification, not reload parity.

The worker uses two Torch CPU threads, one inter-op thread and MPS allocation fraction
0.55. Resource checks at loading/cache/step boundaries record RSS and MPS counters
separately. Stops: 24GiB worker RSS, 22GiB MPS driver allocation, system availability
below25%, swap growth above512MiB, or thirty minutes. Individual long operations can
peak between checks; the separate root supervisor remains required. Never add overlapping
unified-memory counters together.

The helper records actual dataset count/captions; it does not assume the superseded
24-image draft. The current reviewed dataset contains 20 training portraits. The smoke disallows
resume, generated class images, validation inference, Hub publication and checkpoint
pruning. It retains two snapshots plus the final adapter if all twenty steps finish.

Preparation itself has not imported/executed the trainer or established numerical
compatibility. Root owns environment preflight, supervised execution and independent
adapter load/visual evaluation. Twenty successful updates would prove compatibility only,
not successful domain adaptation, final quality or production readiness.

The separately named staged-cache trainer and its revised resource policy are documented
in [CACHED-TRAINER.md](CACHED-TRAINER.md). The original trainer and its 25% availability
guard remain unchanged.
