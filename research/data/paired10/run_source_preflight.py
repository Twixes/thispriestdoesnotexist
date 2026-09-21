"""Strict CPU-only source reproduction check for the approved paired10 manifest."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'manifest-proposed.json'
OUTPUT=HERE/'source-preflight'
EXPECTED='8d0660f1ccf5115a356e9e6b479e4d225c1e9993402fc71575b0fcb9ed8dcff1'
sys.path.insert(0,str(ROOT))
from research.experiments.paired_regions.evaluate import current_source_hashes


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    if OUTPUT.exists() or (HERE/'source-preflight.log').exists():
        raise ValueError('Never overwrite/restart an existing preflight')
    if sha(MANIFEST)!=EXPECTED:
        raise ValueError('Approved manifest checksum mismatch')
    manifest=json.loads(MANIFEST.read_text())
    approval=json.loads((HERE/'root-review.json').read_text())
    if approval['manifest_sha256']!=EXPECTED or approval['mask_research_approved'] is not True:
        raise ValueError('Missing exact root research-mask review')
    prior=json.loads((HERE.parent/'paired7/manifest-proposed-v2.json').read_text())
    if manifest['pairs'][:7]!=prior['pairs']:
        raise ValueError('Earlier seven records changed')
    if len(manifest['pairs'])!=10 or sum(p['split']=='train' for p in manifest['pairs'])!=9 or [p['id'] for p in manifest['pairs'] if p['split']=='validation']!=['028']:
        raise ValueError('Expected nine training pairs and holdout028')
    bundle=(HERE/manifest['source_bundle']).resolve()
    files={MANIFEST,HERE/'root-review.json',bundle/'model.json',bundle/'generator.safetensors'}
    files.update((HERE/row[key]).resolve() for row in manifest['pairs'] for key in ['latent_path','source_path','target_path'])
    before={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)}
    code_before=current_source_hashes()
    pressure=subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True).stdout
    free=int(re.search(r'System-wide memory free percentage: (\d+)%',pressure).group(1))
    (HERE/'source-preflight-memory-before.txt').write_text(pressure)
    if free<25:
        raise RuntimeError(f'Defer: {free}% free memory is below25%')
    command=['/usr/bin/time','-l',str(ROOT/'research/.venv/bin/python'),'-u','-c',
             'import runpy, torch; torch.set_num_interop_threads(1); runpy.run_path("research/experiments/paired_regions/trainer.py", run_name="__main__")',
             '--manifest',str(MANIFEST.relative_to(ROOT)),'--run',str(OUTPUT.relative_to(ROOT)),
             '--preflight-only','--device','cpu','--threads','1','--source-max-uint8-error','0','--w-atol','0']
    record={'command':command,'cwd':str(ROOT),'manifest_sha256':EXPECTED,'source_files_before':code_before,
            'artifact_hashes_before':before,'free_memory_before_percent':free,'interop_threads':1,
            'cpu_threads':1,'backward_or_optimizer_updates':0,'production_approved':False,
            'wrapper_sha256':sha(Path(__file__))}
    invocation=HERE/'source-preflight-invocation.json'
    invocation.write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic()
    with (HERE/'source-preflight.log').open('w') as log:
        process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record.update(exit_code=process.returncode,elapsed_seconds=time.monotonic()-started)
    after={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)}
    code_after=current_source_hashes()
    record.update(artifact_hashes_after=after,source_files_after=code_after,
                  artifacts_unchanged=before==after,code_unchanged=code_before==code_after)
    pressure_after=subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True).stdout
    (HERE/'source-preflight-memory-after.txt').write_text(pressure_after)
    record['free_memory_after_percent']=int(re.search(r'System-wide memory free percentage: (\d+)%',pressure_after).group(1))
    if process.returncode==0:
        result=json.loads((OUTPUT/'preflight.json').read_text())
        assert result['passed'] and result['source_state_unchanged']
        assert len(result['pairs'])==10 and all(p['source_uint8_max_error']==0 and p['w_max_error']==0 for p in result['pairs'])
        record.update(peak_process_rss_bytes=result['peak_process_rss_bytes'],verification_seconds=result['verification_seconds'],
                      all_ten_source_png_and_w_exact=True,source_state_unchanged=True)
    invocation.write_text(json.dumps(record,indent=2)+'\n')
    if OUTPUT.exists():
        (OUTPUT/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
    if before!=after or code_before!=code_after:
        raise RuntimeError('Preflight inputs or immutable code changed')
    print(json.dumps({key:record[key] for key in ['exit_code','elapsed_seconds','free_memory_before_percent','free_memory_after_percent','artifacts_unchanged','code_unchanged']},indent=2))
    return process.returncode


if __name__=='__main__':
    sys.exit(main())
