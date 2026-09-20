# CPU versus MPS gradient audit

The bounded comparison found no evidence of a large Apple GPU backward error in the tested upstream StyleGAN2 operations. This does not establish that the training configuration will converge or that every MPS operation and input behaves identically.

Using PyTorch 2.14.0 and the same trusted NVIDIA FFHQ 256 checkpoint, the script evaluates one deterministic latent and one collar-aligned real image on CPU and MPS. Both models use training mode, FP32, non-fused modulated convolution, constant synthesis noise, no augmentation, and unchanged weights. CPU-generated fake pixels are reused identically for the separate discriminator comparison. The generator comparison includes the complete generator-to-discriminator adversarial gradient path. R1 is computed separately with its actual second-order backward pass; there are no optimizer updates. Parameter gradients are compared for every named trainable parameter.

| Comparison | Aggregate relative L2 error | Cosine similarity | Compared tensors |
| --- | ---: | ---: | ---: |
| Generator adversarial gradients | 0.00075713 (0.0757%) | 0.9999997142 | 110 |
| Discriminator logistic gradients | 0.00002474 (0.00247%) | 0.9999999997 | 38 |
| R1 discriminator parameter gradients | 0.00000535 (0.000535%) | 0.99999999999 | 37 |
| Generated pixels | 0.00000346 | 0.999999999994 | 196,608 values |
| R1 image-input gradients | 0.00074618 (0.0746%) | 0.9999997216 | 196,608 values |

All compared gradients are finite. The discriminator's output bias has no R1 gradient on either backend, as expected: the image derivative removes the additive output bias. No other parameter gradient is missing. Generator loss differs by approximately 0.00000954; discriminator logistic losses are equal at reported float precision; the R1 scalar differs by approximately 0.000000715.

The largest per-tensor generator relative error is 2.23% for `synthesis.b16.conv0.noise_strength`, a scalar whose CPU gradient magnitude is 0.42155. Several other small generator tensors differ by roughly 0.5–1%. These differences are visible in the full report and are not described as bitwise agreement. Overall gradient directions and magnitudes agree closely. The largest generated-pixel absolute difference is 0.0000380 in normalized [-1, 1] units, equivalent to approximately 0.00485 of one 8-bit pixel level. This scale does not resemble the conspicuous double-face artifacts observed in training samples.

Measured phase computation took 3.23 seconds on CPU with two threads and 9.42 seconds on MPS, including first-use GPU overhead and concurrent main-training contention. Total measured comparison time was 12.66 seconds, excluding initial checkpoint loading. Peak process RSS was 2.08 GB; end-of-check MPS allocated memory was 0.297 GB and driver allocation 1.37 GB. These are diagnostic measurements, not isolated throughput benchmarks or peak GPU-memory estimates. The process exited and released its allocations.

Scope limits: batch size one does not exercise cross-sample minibatch statistics, and this check intentionally excludes random noise, augmentation, style mixing, Adam state, FreezeD reconstruction, and accumulated training drift. It uses the pretrained checkpoint, not every subsequent model state. CPU and MPS reduction and activation-boundary differences need not be zero. This result makes a severe general backward failure less likely; it does not identify the remaining cause of slow adaptation or approve any generated model.

`compare.py` is reproducible with `research/.venv/bin/python research/reviews/cpu-mps-gradients/compare.py`. `results.json` contains every tensor's norm, error, cosine, base/input/script hashes, losses, timing, and memory observations. `run.log` preserves the execution summary. No active trainer or dataset was changed.
