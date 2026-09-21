# Native pretrained GAN baseline attempt

The requested comparison did not complete. Official StyleGAN3-T FFHQ1024 produced one
native 1024px warmup, then the 12GiB process-group RSS gate stopped the run before the eight
evaluation samples. StyleGAN2 matching was planned but never reached.

- v1: official runtime failed before inference because setuptools 84 removed pkg_resources.
  The complete failure log and unfiltered latent archive are retained.
- v2: an archived, SHA-pinned setuptools 80.9.0 dependency was added to this worker's import
  path. No shared virtual environment or official NVIDIA source was modified. The correction
  and PyPI wheel provenance are retained.
- v2 terminated with signal 9 after 17.462 seconds; maximum sampled process-group RSS was
  12,986,810,368bytes (12.09GiB). The supervisor sampled every 100ms and killed the complete
  process group; this is a monitored RSS ceiling with observation overshoot, not an OS
  address-space allocation cap. Available-system-memory gate was 20%; timeout 600 seconds.
- Saved warmup forward 15.313 seconds, CPU, one thread FP32, psi 0.7. This is neither warmed steady-state
  latency nor CUDA latency. Final conversion/PNG+WebP encoding took 0.166 seconds.

The warmup is a coherent male-appearing adult face with closed eyes, sunglasses on the head,
and a tight crop showing bare neck with virtually no shirt/chest. Native 1024px was visually
inspected and both output hashes verified. It is one predetermined warmup, insufficient for
identity diversity or acceptance-rate conclusions, and it is not a trained priest.

Artifacts: `research/runs/pretrained-gan-baseline8-v1/` and
`research/runs/pretrained-gan-baseline8-v2/`. Checkpoint/source archives are in
`research/models/stylegan3-t-ffhq1024/`; unchanged official runtime is in
`research/vendor/stylegan3/`. Images, latent arrays, checkpoint, source archive and dependency
wheel are covered by Git LFS. All original failed artifacts remain.

No resource-gate retry was made. A CUDA comparison with native filtered_lrelu kernels is
needed before rejecting this base on server latency grounds. No production approval or
sub-500ms claim follows from this attempt.
