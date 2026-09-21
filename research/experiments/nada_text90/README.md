# Matched text-direction probe: total10 →100

Prepared only. No models loaded, subprocess lifecycle fixtures started, or training executed. The active image-guided run and its pinned files are unchanged. `review.diff` is the complete source delta from `nada_clean24_continue500`.

This changes **only the detached guidance direction**, plus the declared90-update horizon/checkpoint schedule. The same checkpoint010 student/Adam/RNG, same first90 continuation training z, same eight evaluation z,24 target provenance, source generator, image preprocessing, trainable layers, LR, cosine loss and per-step source forward remain. There are140 unique original50+fresh90 z rows. This is a matched contrast, not a new random-seed search.

The existing pinned ViT-B/32 text encoder computes normalized source and target embeddings for four fixed photo-only templates:

- `a photo of {}.`
- `a portrait photo of {}.`
- `a black and white photo of {}.`
- `a close-up photo of {}.`

Source: `a man wearing an ordinary shirt`. Target: `a man wearing a black clerical shirt with a white Roman collar`.

The direction is `unit(mean(unit(Etext(target_i)) - unit(Etext(source_i))))`, under no-grad. It replaces the prior image-centroid direction; no global, patch or auxiliary term is added. Encoding must leave CLIP state and training RNG unchanged. `text-direction.json` records exact prompts, embeddings, direction, norm and hash. The source/target classes deliberately share “a man” and differ in clothing. Photo-only templates are an explicit domain choice; this is not a guarantee CLIP understands Roman-tab geometry.

Before update11, the inherited actual checkpoint/Adam/RNG restoration gate and all16 native step10 RGB/grayscale PNG hash matches must pass. No bootstrap repeats. The image-gradient, finite, frozen-state, snapshot-RNG and actual checkpoint-reload checks are unchanged. Full atomic checkpoint/previews at total50 and100 enable matched review against the existing image-guided artifacts. The goal is recognizable collar/shirt geometry without worsening face detail; lower CLIP loss, darker clothes or stronger eyebrows alone fail. Children/hats are not guaranteed to disappear.

Resource policy: CPU1 FP32; fresh35% free-memory checks;12GiB sampled RSS;20-minute absolute deadline including setup; no competing heavy research jobs; no automatic retry. Only the owned child group can be terminated/reaped. Source-feature overhead remains measured per update. The shorter horizon is the only supervision-policy change from the60-minute500-total run.

Four lightweight checks passed:265 actual file pins/no model imports; exact matched data; unchanged AST for image/loss-update/resume/snapshot helpers; fixed prompts and source compilation. **No runner command, even check-only, and no lifecycle subprocess test was invoked while the other NADA run was active.** Actual text encoding and restored native equality remain execution gates. The lifecycle implementation is inherited; this preparation intentionally did not rerun its child fixtures concurrently.

After root confirms the previous model process is terminal, optional static provenance check:

```sh
research/selection/.venv/bin/python research/experiments/nada_text90/runner.py --check-only
```

Proposed actual launch, not executed:

```sh
research/selection/.venv/bin/python -u research/experiments/nada_text90/runner.py \
  --execute --output research/runs/nada-text90-cpu-matched100
```

Output must be new. Normal execution includes the initial restoration gate; the inherited optional restore-only mode merely checks restoration and stops before computing text guidance or updating. No production promotion or model-quality approval is implied.
