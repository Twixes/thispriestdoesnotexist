#!/usr/bin/env python3
"""Retrieve only the official NVIDIA checkpoint; do not execute pickle content."""
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent
URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl'
TARGET = BASE / 'stylegan3-t-ffhq-1024x1024.pkl'
EXPECTED_BYTES = 294775112
EXPECTED_SHA256 = 'efd9fa1f967a11b5390399a8ed512dc32e341c245fa2d4dafa6d94e96222085b'
with urllib.request.urlopen(URL, timeout=60) as response:
    require_length = response.headers.get('Content-Length')
    if require_length and int(require_length) != EXPECTED_BYTES:
        raise RuntimeError('Unexpected official checkpoint byte length')
    digest = hashlib.sha256()
    count = 0
    temporary = TARGET.with_suffix('.partial')
    with temporary.open('wb') as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > EXPECTED_BYTES:
                raise RuntimeError('Download exceeded expected size')
            digest.update(chunk)
            output.write(chunk)
    if count != EXPECTED_BYTES:
        raise RuntimeError('Incomplete official checkpoint')
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Official checkpoint bytes changed from pinned download')
    temporary.replace(TARGET)
    provenance = {
        'model': 'NVIDIA StyleGAN3-T FFHQ 1024x1024',
        'source_url': URL,
        'official_listing': 'https://github.com/NVlabs/stylegan3/blob/c233a919a6faee6e36a316ddd4eddababad1adf9/README.md',
        'source_code_commit': 'c233a919a6faee6e36a316ddd4eddababad1adf9',
        'ngc_version': '1', 'sha256': digest.hexdigest(), 'bytes': count,
        'etag': response.headers.get('ETag'),
        'upstream_object_version': response.headers.get('x-amz-version-id'),
        'license': 'NVIDIA Source Code License for StyleGAN3; research/evaluation only under section3.3; retain notices',
        'license_path': 'research/models/stylegan3-t-ffhq1024/LICENSE.txt',
        'declared_resolution': 1024,
        'architecture_loaded_or_executed': False,
        'downloaded_on': '2026-09-21',
        'production_approved': False,
        'purpose': 'Distinct pretrained base candidate for bounded native-quality/framing/adaptation comparison'
    }
    (BASE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))
