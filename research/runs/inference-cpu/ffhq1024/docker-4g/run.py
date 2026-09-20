"""Run one cold and three warm forwards under a Docker-enforced memory cap."""
import json
from pathlib import Path
import subprocess
import sys
import time

output = Path('/results')
started = time.monotonic()
result = subprocess.run([
    sys.executable, '-m', 'inference.benchmark', '--model-dir', '/app/model',
    '--output', str(output), '--threads', '1', '--count', '4', '--allow-unreviewed',
])
elapsed = time.monotonic() - started
metrics = {
    'exit_code': result.returncode,
    'elapsed_seconds': elapsed,
    'cpu_limit': 1,
    'memory_limit_bytes': 4294967296,
    'network': 'none',
    'cgroup_memory_peak_bytes': int(Path('/sys/fs/cgroup/memory.peak').read_text()),
    'cgroup_memory_events': Path('/sys/fs/cgroup/memory.events').read_text(),
    'cgroup_cpu_stat': Path('/sys/fs/cgroup/cpu.stat').read_text(),
    'interpretation': 'Linux amd64 emulation on Apple Silicon; memory evidence only, not production latency.',
}
benchmark = output / 'benchmark.json'
if benchmark.exists():
    report = json.loads(benchmark.read_text())
    metrics['process_start_to_first_sample_seconds_approx'] = elapsed - sum(
        row['seconds'] for row in report['generations'][1:])
(output / 'container-metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics), flush=True)
sys.exit(0 if result.returncode == 0 else 1)
