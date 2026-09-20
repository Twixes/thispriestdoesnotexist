# CDC next experiment after the aligned step-500 review

## Recommendation

Start the CDC arm from **the original FFHQ base**, using the same seed, aligned 110-image dataset and all other settings as `aligned110-frozen4`. Keep that run as the zero-CDC control and compare at matched training steps/image presentations. Add only `--cdc-weight 1000 --cdc-batch 4`. Do not use the newly created resume-test checkpoint as a research starting point; it is a batch-1 plumbing test on the unaligned smoke dataset.

The parent visually reviewed aligned step 500 and reported repeated narrow faces/hair with ghosted eyes and mouths despite recognizable collars. This is a preservation problem, not just a missing collar. CDC constrains changes to relationships already present in the source. Applying it after severe contraction tests recovery from a damaged initialization, which is a different and harder question. A paired continuation from the same step-500 checkpoint would isolate recovery effects, but it is not the most useful next quality experiment. Preserve that checkpoint for a later recovery ablation if needed.

The fresh control/CDC comparison shares pretrained initialization, seed, corpus and optimization settings. The dedicated CDC latent generator leaves global CPU/MPS RNG untouched (verified), so it does not intentionally change the adversarial random draws. Training trajectories will still diverge because CDC changes G and consequently discriminator feedback/augmentation adaptation; matching the initial seed does not mean every later state will be identical.

Parent launched the following fresh-base CDC arm as `aligned110-frozen4-cdc1000` after the resume test passed. The first 10 updates were finite in 38.4 seconds while the control continued (parent-reported launch evidence). No quality conclusion is available from those early losses. The subagent did not launch it:

```sh
research/.venv/bin/python -u research/train.py \
  --device mps --threads 8 --batch 8 --steps 6000 --snapshot-every 250 \
  --base research/models/ffhq256.pkl \
  --data research/alignment/collar-only/eyes42 \
  --mapping-lr 0.0005 --synthesis-lr 0.001 --d-lr 0.001 \
  --ema-kimg 0.5 --ada-kimg 100 --augment-p 0.5 --r1-gamma 2 \
  --freeze-d-layers 4 --cdc-weight 1000 --cdc-batch 4 \
  --run aligned110-frozen4-cdc1000
```

Confirm the dataset checksum matches the control: `0f5c978c3c153b3943396c6488c3a371fd5d0fc5e2be1a36311d8cd808b06688`. Weight 1,000 is a conservative initial calibration, not an established optimal weight: on the old pilot its global gradient was 4.65% of the adversarial gradient, but its mapping contribution was already 3.07 times the adversarial mapping gradient. Log per-block and weighted losses; recalibrate if correspondence prevents adult male/collar adaptation.

At 250/500/1,000, compare matched raw G and untruncated EMA (`psi=1`) alongside the existing truncated preview. Use the same held-out latent set for both arms; inspect facial integrity, distinct identities/hair/age, collar coherence and hats at display size. A preserved generic FFHQ face with no priest adaptation is not success, nor is a sharp but repeated priest. Quality and diversity trends decide whether extending the run is justified. No short-run numeric loss establishes deployment quality.

## Training duration interpretation

NVIDIA's official README says **1,000 kimg is often enough for transfer learning**. Kimg counts thousands of real-image presentations to D, not optimizer steps or unique images. Its timing table uses V100 hardware and the official CUDA recipe; those times do not predict this MPS trainer. [Official primary source](https://github.com/NVlabs/stylegan2-ada-pytorch#expected-training-time).

Here 6,000 × batch 8 = **48 kimg**, only 4.8% of that reference; 1,000 kimg at batch 8 would be 125,000 steps. The current 6,000-step cap is therefore an early feasibility/comparison horizon, not a promised converged model. The guidance is neither a hard minimum nor proof that longer training will repair early mode collapse. Our 110-image domain, DiffAugment configuration, missing path-length regularization and CPU/MPS implementation differ from the reference. Extend a preserving/adapting trajectory on evidence; do not spend 125,000 steps on visibly contracting faces merely to reach a quoted count.

## Full resume verification and compute limits

`research/experiments/cdc/check_resume_roundtrip.py` ran two separate real MPS processes at 256px. The first made one G/D update and wrote the trainer's actual checkpoint. The second loaded it and made its next G/D/R1 update. Only preview-grid generation was skipped; the training and checkpoint paths were unchanged.

- All G Adam entries (110) and D Adam entries (31), including moment tensors and parameter groups, matched the serialized values exactly after loading.
- CDC config, base/data checksums, learning rates, FreezeD setting, step and image counter matched.
- The saved independent CDC RNG state and exact next latent batch matched.
- Frozen source remained the original raw FFHQ G, distinct from resumed target G, with no gradients or state changes.
- Both updates and resumed R1 were finite; frozen D remained bitwise unchanged. Resumed CDC was nonzero (`5.5922e-8`).

Evidence: `resume-roundtrip-create.json`, `resume-roundtrip-resume.json` and matching logs under `research/experiments/cdc/`; actual checkpoints/configs under `research/runs/cdc-resume-roundtrip-*`. No trainer bug or edit was required. This checks checkpoint restoration plus a valid next update, not bitwise equality to a separate uninterrupted MPS run.

The resumed batch-1 adversarial / batch-4 CDC test used about **3.97 GB driver allocation** after its update; the previous standalone CDC batch-4 forward/backward took **1.55 seconds**. These are bounded process snapshots while the main run continued. They do not certify the peak for simultaneous batch-8 jobs. Parent should watch actual allocator/host pressure and iteration slowdown during the concurrent run, rather than extrapolating linearly. Parent has launched the run as described above; no further diagnostic compute is pending.
