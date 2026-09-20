"""Bounded research-only real HTTP test entirely inside an isolated container."""
import concurrent.futures
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image

OUT = Path('/results')
assert os.environ.get('ALLOW_UNREVIEWED') == '1', 'Explicit research override required'
EXPECTED = json.loads(Path('/app/model/model.json').read_text())['weights_sha256']
COMMAND = [sys.executable, '-m', 'inference.server', '--host', '127.0.0.1', '--port', '8080',
           '--threads', '1', '--capacity', '3', '--allow-unreviewed']

def metrics():
    results = {}
    for name in ['memory.current', 'memory.peak', 'memory.max', 'memory.swap.max', 'memory.events', 'cpu.stat', 'pids.current']:
        path = Path('/sys/fs/cgroup') / name
        results[name] = path.read_text().strip() if path.exists() else None
    return results

def request(path):
    started = time.monotonic()
    try:
        response = urllib.request.urlopen('http://127.0.0.1:8080' + path, timeout=120)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, dict(response.headers), response.read(), time.monotonic() - started

def verify_image(label, result):
    status, headers, body, seconds = result
    normalized = {k.lower(): v for k, v in headers.items()}
    assert status == 200, (label, status)
    assert normalized['content-type'] == 'image/webp'
    assert 'no-store' in normalized['cache-control']
    assert normalized['x-model-sha256'] == EXPECTED
    assert re.fullmatch('[0-9a-f]{64}', normalized['x-generation-seed'])
    with Image.open(io.BytesIO(body)) as image:
        image.load()
        assert image.size == (1024, 1024) and image.mode == 'RGB' and image.format == 'WEBP'
    target = OUT / (label + '.webp')
    target.write_bytes(body)
    record = {'label': label, 'status': status, 'headers': headers, 'http_elapsed_seconds': seconds,
              'sha256': hashlib.sha256(body).hexdigest(), 'seed': normalized['x-generation-seed'],
              'decoded_size': [1024, 1024], 'decoded_mode': 'RGB', 'bytes': len(body)}
    print(json.dumps(record), flush=True)
    return record

report = {'started_unix': time.time(), 'server_command': COMMAND,
          'platform': platform.platform(), 'machine': platform.machine(),
          'model_sha256': EXPECTED, 'local_research_override': True,
          'server_source_sha256': hashlib.sha256(Path('/app/inference/server.py').read_bytes()).hexdigest(),
          'metrics_before': metrics(), 'images': [], 'overload': [], 'passed': False}
started = time.monotonic()
server_log = (OUT / 'server.log').open('w')
server = subprocess.Popen(COMMAND, stdout=server_log, stderr=subprocess.STDOUT)
held = []
try:
    deadline = time.monotonic() + 60
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited before readiness: {server.returncode}')
        try:
            status, headers, body, elapsed = request('/health')
            if status == 200:
                health = json.loads(body)
                assert health['status'] == 'ready' and health['resolution'] == 1024
                assert health['model_sha256'] == EXPECTED and health['training_step'] == 0
                assert health['reviewed'] is False and health['device'] == 'cpu'
                assert 'no-store' in headers['Cache-Control']
                report['health'] = {'body': health, 'headers': headers}
                break
        except (urllib.error.URLError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('Server did not become healthy')
        time.sleep(.2)
    report['process_launch_to_health_seconds'] = time.monotonic() - started
    for index in range(2):
        report['images'].append(verify_image(f'sequential-{index}', request(f'/generate?nonce=sequential-{index}')))

    # Three incomplete headers deterministically occupy admission slots without
    # changing the server or mocking generation. Finish all three after probing
    # overload, then allow the real model to serve those requests serially.
    baseline_threads = len(list(Path(f'/proc/{server.pid}/task').iterdir()))
    for index in range(3):
        connection = socket.create_connection(('127.0.0.1', 8080), timeout=120)
        connection.sendall(f'GET /generate?nonce=concurrent-{index} HTTP/1.1\r\nHost: localhost\r\n'.encode())
        held.append(connection)
    admission_deadline = time.monotonic() + 3
    while len(list(Path(f'/proc/{server.pid}/task').iterdir())) < baseline_threads + 3:
        if time.monotonic() >= admission_deadline:
            raise TimeoutError('Three requests were not admitted promptly')
        time.sleep(.02)
    report['admission_threads'] = {'baseline': baseline_threads,
                                  'with_three_held_requests': len(list(Path(f'/proc/{server.pid}/task').iterdir()))}
    for path in ['/generate?nonce=overload', '/health']:
        status, headers, body, elapsed = request(path)
        assert status == 503 and body == b''
        assert headers['Retry-After'] == '3' and 'no-store' in headers['Cache-Control']
        report['overload'].append({'path': path, 'status': status, 'headers': headers, 'seconds': elapsed})
    released = time.monotonic()
    for connection in held:
        connection.sendall(b'\r\n')
    def read_held(index):
        with http.client.HTTPResponse(held[index]) as response:
            response.begin()
            body = response.read()
            return index, (response.status, dict(response.headers), body, time.monotonic() - released)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for index, result in pool.map(read_held, range(3)):
            report['images'].append(verify_image(f'concurrent-{index}', result))
    for connection in held:
        connection.close()
    held = []
    status, headers, body, elapsed = request('/health')
    assert status == 200 and json.loads(body)['model_sha256'] == EXPECTED
    report['recovery_health'] = {'status': status, 'headers': headers, 'seconds': elapsed}
    report['images'].append(verify_image('recovery', request('/generate?nonce=recovery')))
    assert len({item['seed'] for item in report['images']}) == 6
    assert len({item['sha256'] for item in report['images']}) == 6
    report['metrics_after_requests'] = metrics()
    report['server_proc_status_before_stop'] = Path(f'/proc/{server.pid}/status').read_text()
    report['passed'] = True
except BaseException as error:
    report['error'] = {'type': type(error).__name__, 'message': str(error)}
    raise
finally:
    for connection in held:
        connection.close()
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait(timeout=10)
            report['server_forced_kill'] = True
    report['server_exit_code'] = server.returncode
    server_log.close()
    report['total_seconds'] = time.monotonic() - started
    report['metrics_after_server_exit'] = metrics()
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'server_exit_code': server.returncode,
                      'total_seconds': report['total_seconds'], 'metrics': metrics()}), flush=True)
if server.returncode != 0:
    raise SystemExit(server.returncode)
