"""Local resource-cap experiment; never a production deploy or quality approval."""
import json
from pathlib import Path
import runpy
import sys
import time

started = time.monotonic()
sys.argv = ["inference.benchmark", "--model-dir", "/app/model", "--output", "/results",
            "--threads", "1", "--count", "4", "--allow-unreviewed"]
runpy.run_module("inference.benchmark", run_name="__main__")
report = json.loads(Path("/results/benchmark.json").read_text())
finished = time.monotonic()
report["resource_simulation"] = {
    "cpu_limit": 1,
    "memory_limit_bytes": 3221225472,
    "cpu_threads": 1,
    "process_start_to_all_four_samples_seconds": finished - started,
    "process_start_to_first_sample_seconds_approx": finished - started
        - sum(row["seconds"] for row in report["generations"][1:]),
    "cgroup_memory_peak_bytes": int(Path("/sys/fs/cgroup/memory.peak").read_text()),
    "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
    "cgroup_cpu_stat": Path("/sys/fs/cgroup/cpu.stat").read_text(),
    "platform_limit": "Docker linux/amd64 under Apple Silicon emulation; not Cloudflare hardware",
}
Path("/results/benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["resource_simulation"]), flush=True)
