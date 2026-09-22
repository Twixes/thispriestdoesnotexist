# Exact continuation after main250

`continue_main_adaptation.py` changes only the continuation horizon, snapshot steps and explicit wall deadline. It imports the frozen `AdaptationEngine` and existing authenticated checkpoint restore helper. It preserves the training images, masks, original protected references, all G/D/GEMA tensors, Adam states, both RNGs, path target, optimizer parameter order, absolute regularizer schedule, CPU FP32 and one thread.

The first input must be the **completed** `research/runs/adam-native1024-retained250-main100-to250-v1`, with its pinned protocol `2c10cfc0ebec3ab0568d2778f3fd5363d231e52931d81ce4522cc4660b038c33`. Later inputs must be completed endpoints produced by this exact runner source. Validation traces parent links to that bootstrap, verifies source/configuration/file hashes and completed supervisor/result/checkpoint records. The real checkpoint250 hash is read only after completion; none is assumed here. A failed run or an intermediate snapshot is not an accepted source.

Before each continuation updates any weights, it reconstructs original immutable references, restores the full source checkpoint, verifies the complete state and reproduces all eight parent raw/EMA PNGs byte for byte. Each requested endpoint records four fixed-latent raw and four EMA images, serializes and restores a full checkpoint, and the terminal endpoint also verifies exact raw/EMA folding. Owned worker supervision retains 35% available memory at start, 20% while running, 12 GiB RSS, 512 MiB swap-growth and the requested wall deadline. There are no automatic retries or launches.

These are **conditional future commands**, not evidence that main250 completed or that any subsequent endpoint passed visual review. Run the next stage only after root has reviewed the preceding endpoint. The suggested deadlines include overhead beyond the observed approximately 14 seconds per update.

```sh
research/.venv/bin/python research/experiments/adam_native/continue_main_adaptation.py \
  --run research/runs/adam-native1024-retained250-main100-to250-v1 \
  --until 500 --snapshots 500 --seconds 5400 \
  --output research/runs/adam-native1024-retained250-main250-to500-v1

research/.venv/bin/python research/experiments/adam_native/continue_main_adaptation.py \
  --run research/runs/adam-native1024-retained250-main250-to500-v1 \
  --until 1000 --snapshots 1000 --seconds 10800 \
  --output research/runs/adam-native1024-retained250-main500-to1000-v1

research/.venv/bin/python research/experiments/adam_native/continue_main_adaptation.py \
  --run research/runs/adam-native1024-retained250-main500-to1000-v1 \
  --until 1500 --snapshots 1500 --seconds 10800 \
  --output research/runs/adam-native1024-retained250-main1000-to1500-v1
```

Bounded verification: `research/.venv/bin/python research/experiments/adam_native/test_continue_main_adaptation.py`. Four tests cover full tiny-engine uninterrupted/staged equality across regularization boundaries, invalid checkpoint-envelope rejection before mutation, schedules, and explicitly opaque synthetic metadata fixtures that authenticate both accepted source formats and reject incomplete, altered or unknown inputs. These fixtures are not native results. No native model is loaded by the tests.

Longer training remains an experiment: 1500 batch-one updates still expose fewer real samples than the source batch-four recipe. Grayscale drift alone is expected from the grayscale target dataset; actual domain conversion, facial realism, diversity and forbidden hats/minors require separate image review. No endpoint here is production-approved.
