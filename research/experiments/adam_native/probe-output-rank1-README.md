# Prepared output-rank-one probing variant

`probe_output_rank1.py` is prepared and **has not been launched**. It derives from frozen `probe.py` SHA-256 `f77541904b211a60ca6be24f89e22153c63a0d74afd8fb37fbb638191e5d28a1`. Neither that source nor `modulation.py` was changed.

The only training change is passing `conv_layout='output_rank1'` explicitly to both G and D `install_modulation` calls. For convolution weights, the existing helper uses `outer(v_vector, u_vector)` reshaped into `[output, input, height, width]`. The control defaults to `source_flattened`, which uses the reverse outer-product order before reshaping. Fully connected weights already use `output_rank1` in both variants.

The worker AST is identical to the frozen source after removing exactly those two keyword arguments. The supervisor AST is identical after normalizing the executed filename. The variant preserves 500 iterations, checkpoints at 100/250/500, full model/optimizer/RNG/path-state restoration checks, all losses and optimizer settings, source weights, training examples, seeds, and every memory/time guard. Metadata changes identify the new protocol, explicit layout, source hash, executed filename and source archives. `probe-output-rank1-preparation.json` retains hashes and the complete source diff.

Launch only after the correct-layout smoke has actually succeeded and the current control run is terminal, with root coordinating compute. The prerequisites are scheduling requirements; the CLI itself does not inspect other running processes or smoke results. Use a fresh output directory:

```sh
research/.venv/bin/python research/experiments/adam_native/probe_output_rank1.py \
  --output research/runs/adam-native1024-output-rank1-probing500-v1
```

The variant checks the original probe hash at preparation, pins both source files and the unchanged modulation helper, and archives its executed source as `probe_output_rank1.py` alongside `source-probe.py`. Static parsing, compilation without execution, and AST comparisons passed. No model was loaded or trained during preparation. This remains importance probing: no Fisher estimation, main adaptation, convergence, priest-generation quality acceptance, or production approval is implied.
