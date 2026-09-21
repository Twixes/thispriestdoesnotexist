# Convolution layout decision

The current `adam-native1024-probing500-v1` run is a control following the released source's convolution layout. It is not a completed paper-aligned importance-selection experiment. Its running code and objective remain frozen.

The [paper, pages 6–8](https://openreview.net/pdf/44f72b6c163a1eaea103348545b78dfbc262d4d5.pdf) defines each convolution kernel by its output channel. Its modulation uses an outer product of an output-channel vector with a flattened input/kernel vector. Thus one output-vector entry controls one kernel. Main adaptation preserves important base kernels through restricted modulation and directly trains the other kernels.

Our independently checked source uses the reverse outer-product order before reshaping. With two outputs and four flattened inputs, its rows are `[u0*v0, u0*v1, u1*v0, u1*v1]` and `[u2*v0, u2*v1, u3*v0, u3*v1]`. Both rows depend on both output-vector entries. A source-style mask on `v0` therefore does not isolate original weight row zero.

Decision: keep this source-layout run as an explicitly labeled control. For literal per-output kernel importance and selective adaptation, use the existing `output_rank1` implementation in a separate, fresh-source, pinned experiment and recompute importance there. Do not transplant or transpose the current run's learned factors or scores. Keep source-model, data, seeds, native resolution, optimizer, losses, and resource limits matched for the comparison. This changes training parameterization; inference remains an ordinary unconditional generator after folding.

Before launching that separate experiment, validate its native initial equivalence, trainable-factor gradients, frozen weights, regularizers and exact folded export. A successful source-layout control does not prove those outcomes for the alternative. Neither finite losses nor either layout guarantees photographic quality or adult priest appearance.
