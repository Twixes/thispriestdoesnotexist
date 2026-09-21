"""Read-only CPU validation of trusted local step-500 training artifacts."""
import hashlib
import json
import math
import os
import shutil
import re
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    preflight = subprocess.check_output(['memory_pressure'], text=True)
    (RUN/'checkpoint-000500-memory-before.txt').write_text(preflight)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', preflight)[1])
    if free < 35:
        raise SystemExit('Defer: less than 35% memory free; no checkpoint load')
    candidate = RUN/'resume-000500-candidate.pt'
    source_resume = candidate if candidate.exists() else RUN/'resume.pt'
    candidate_metadata = None
    if candidate.exists():
        candidate_metadata = json.loads((RUN/'resume-000500-candidate.json').read_text())
        assert candidate_metadata['status'] == 'UNVALIDATED_CANDIDATE'
        assert candidate_metadata['checkpoint_step_verified'] is False
        assert digest(candidate) == candidate_metadata['candidate_sha256']
    paths = [source_resume] + [RUN/name for name in ['generator-000500.pt','config.json',
             'train-source.py','adapters-source.py','samples-000500-ema.png',
             'samples-000500-raw.png','samples-000500-untruncated.png']]
    before = {p.name:digest(p) for p in paths}
    started = time.perf_counter()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Trainer schema stores NumPy/Python/MPS RNG and statistics objects in the
    # full local resume. It explicitly uses weights_only=False when resuming.
    # This file is our own trusted output, never an external downloaded pickle.
    checkpoint = torch.load(source_resume, map_location='cpu', mmap=True, weights_only=False)
    # The dedicated generator export was deliberately made weights-only safe.
    generator = torch.load(RUN/'generator-000500.pt', map_location='cpu', mmap=True, weights_only=True)
    config = json.loads((RUN/'config.json').read_text())
    assert checkpoint['format_version'] == 1
    assert checkpoint['batch_idx'] == generator['step'] == 500
    assert checkpoint['images_seen'] == generator['images_seen'] == 500*64 == 32000
    assert checkpoint['sampler']['consumed'] == 32000
    assert checkpoint['sampler']['seed'] == config['recipe']['seed']
    assert checkpoint['recipe'] == generator['recipe'] == config['recipe']
    assert checkpoint['config'] == generator['config'] == config
    assert config['trainer_sha256'] == before['train-source.py']
    tensors = []
    invalid = []
    def walk(value, path):
        if torch.is_tensor(value):
            assert value.device.type == 'cpu', path
            # All saved tensors are contiguous here. Refuse a hidden reshape
            # clone rather than allocate a second large weight tensor.
            assert value.is_contiguous(), path
            flat = value.view(-1)
            finite = True
            if value.is_floating_point() or value.is_complex():
                for start in range(0, flat.numel(), 262144):
                    if not bool(torch.isfinite(flat[start:start+262144]).all()):
                        finite = False
                        break
            tensors.append({'path':path,'shape':list(value.shape),'dtype':str(value.dtype),
                            'elements':value.numel(),'finite':finite})
            if not finite:
                invalid.append(path)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, path+'/'+str(key))
        elif isinstance(value, (tuple,list)):
            for i,item in enumerate(value):
                walk(item, path+'/'+str(i))
    walk(checkpoint, 'resume')
    walk(generator, 'generator')
    assert not invalid, invalid
    assert set(checkpoint['G_ema']) == set(generator['G_ema'])
    comparisons = []
    for key, left in checkpoint['G_ema'].items():
        right = generator['G_ema'][key]
        assert left.shape == right.shape and left.dtype == right.dtype
        equal = torch.equal(left, right)
        assert equal, key
        comparisons.append({'name':key,'elements':left.numel(),'equal':equal})
    after = {p.name:digest(p) for p in paths}
    assert before == after, 'Artifact changed while being reviewed; discard result'
    preserved = RUN/'resume-000500.pt'
    temporary = RUN/'resume-000500.pt.copying'
    if preserved.exists() or temporary.exists():
        raise RuntimeError('Refuse to overwrite an existing checkpoint500 preservation artifact')
    shutil.copyfile(source_resume,temporary)
    with temporary.open('rb') as f:
        os.fsync(f.fileno())
    copied_hash = digest(temporary)
    assert copied_hash == before[source_resume.name] == digest(source_resume), 'Live resume changed during preservation'
    os.replace(temporary,preserved)
    directory_fd = os.open(RUN,os.O_RDONLY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    assert digest(preserved) == copied_hash
    result = {
        'reviewed_utc':datetime.now(timezone.utc).isoformat(),
        'validation':'passed', 'production_approved':False,
        'scope':'CPU mmap finite-tensor and exact G_ema export validation; no inference or training',
        'resume_step_field':'batch_idx','resume_batch_idx':checkpoint['batch_idx'],
        'generator_step_field':'step','generator_step':generator['step'],
        'images_seen':checkpoint['images_seen'],'sampler':checkpoint['sampler'],
        'schema_source':'train-source.py:73-82, 250-259, 329-337',
        'load_policy':{'resume':'mmap=True, CPU, weights_only=False: trusted own trainer output with NumPy RNG state',
                       'generator':'mmap=True, CPU, weights_only=True'},
        'torch_version':str(torch.__version__),'threads':torch.get_num_threads(),
        'memory_preflight_free_percent':free,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'duration_seconds':time.perf_counter()-started,
        'finite_tensors':len(tensors),'total_tensor_elements':sum(t['elements'] for t in tensors),
        'nonfinite_tensors':invalid,'ema_equal_tensors':len(comparisons),
        'ema_equal_elements':sum(c['elements'] for c in comparisons),
        'hashes_before':before,'hashes_after':after,'artifacts_unchanged':before==after,
        'validated_resume_input':source_resume.name,
        'provisional_capture_metadata':candidate_metadata,
        'preserved_checkpoint':preserved.name,'preserved_checkpoint_sha256':copied_hash,
        'preserved_copy_matches_validated_source':True,
        'template_validator':'verify-checkpoint-000375.py',
        'template_validator_sha256':digest(RUN/'verify-checkpoint-000375.py'),
        'validator_sha256':digest(Path(__file__)),
        'tensors':tensors,'ema_comparisons':comparisons,
        'limits':['Numeric finiteness and export equality are not image-quality approval.',
                  'Saved grids are checked by hashes and trainer save order; no forward was rerun to reproduce pixels.',
                  'Only step500 artifacts are reviewed; the live training process continued unchanged.',
                  'An exact copy of the validated live resume was preserved separately as resume-000500.pt.'],
    }
    (RUN/'checkpoint-000500-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['tensors','ema_comparisons']},indent=2))


if __name__ == '__main__':
    main()
