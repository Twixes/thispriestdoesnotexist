# Approved paired10 source reproduction preflight

Passed with exit 0. All ten pairs reproduce their stored native source PNG and mapped W exactly on CPU with one intra-op and one inter-op thread, using zero tolerance. The manifest contains nine training identities and unchanged holdout 028. This checks data provenance, not learned quality or production readiness.

Approved manifest SHA256: `8d0660f1ccf5115a356e9e6b479e4d225c1e9993402fc71575b0fcb9ed8dcff1`. The earlier seven records are unchanged. Before/after hashes for all source, target, latent, bundle and approval files match; the immutable trainer/helper/vendor inventory also matches. The source model state remained unchanged. No optimizer or backward operation occurred.

The actual invocation is in `invocation.json`. The unchanged region trainer ran with `--preflight-only`, `--device cpu`, `--threads 1`, `--source-max-uint8-error 0`, and `--w-atol 0`; the wrapper set inter-op threads to one. Total process time was 9.92 seconds, including 8.78 seconds of source verification. Peak native macOS RSS was 4,034,936,832 bytes (3.76 GiB); `/usr/bin/time -l` additionally reported peak memory footprint 4,117,302,824 bytes. These are distinct process metrics, not a hosting estimate.

The fresh memory guard passed at 31% free. The immediate post-process reading was 22%; the process had exited and was absent in the subsequent process check. Root was notified to obtain a fresh guard before another heavy job. No new job or training was launched afterward by this task.

The sibling `run_source_preflight.py` records before/after hashes and refuses to overwrite or restart this result. Full output is `../source-preflight.log`; memory evidence and duplicate launch record are stored alongside it. No dataset, trainer, model weight or approval gate was changed.
