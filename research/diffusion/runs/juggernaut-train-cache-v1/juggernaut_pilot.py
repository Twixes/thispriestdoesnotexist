"""Read-only provenance checks shared by the bounded Juggernaut cache and trainer."""
import json
from pathlib import Path
from compare import ROOT, sha

PROTOCOL = ROOT / 'juggernaut-pilot-protocol.json'
MODEL = ROOT / 'models/juggernaut-x-hyper-components'
DATASET = ROOT.parent / 'data/flux-priest-domain-v1'


def preflight():
    protocol = json.loads(PROTOCOL.read_text())
    source = ROOT / 'models/juggernaut-x-hyper/provenance.json'
    if sha(source) != protocol['base']['source_provenance_sha256']:
        raise ValueError('Juggernaut publisher provenance changed')
    if sha(DATASET/'manifest.json') != protocol['dataset']['manifest_sha256']:
        raise ValueError('Reviewed dataset manifest changed')
    cases = ROOT.parents[1] / protocol['development_cases']['manifest_path']
    if sha(cases) != protocol['development_cases']['manifest_sha256']:
        raise ValueError('Frozen hot-priest development cases changed')
    if sha(MODEL/'provenance.json') != protocol['base']['components_provenance_sha256']:
        raise ValueError('Frozen component provenance changed')
    manifest = json.loads((MODEL/'provenance.json').read_text())
    if not manifest['complete'] or manifest['model'] != protocol['base']['model_id'] or manifest['revision'] != protocol['base']['revision']:
        raise ValueError('Incomplete or different Juggernaut component export')
    if manifest['upstream_checkpoint_sha256'] != protocol['base']['checkpoint_sha256'] or manifest['upstream_provenance_sha256'] != sha(source):
        raise ValueError('Component export has different source checkpoint')
    verified = {v['component']:v for v in manifest['verifications']}
    for name in ('unet','vae','text_encoder','text_encoder_2'):
        if name not in verified or not verified[name]['equal'] or verified[name]['tensor_count'] <= 0:
            raise ValueError('Component roundtrip not verified: '+name)
    for file in manifest['files']:
        path = (MODEL/file['path']).resolve()
        if not path.is_relative_to(MODEL.resolve()) or path.stat().st_size != file['bytes'] or sha(path) != file['sha256']:
            raise ValueError('Component file missing or changed; use exporter restore tool: '+file['path'])
    if sha(MODEL/'scheduler/scheduler_config.json') != protocol['base']['scheduler_config_source_sha256']:
        raise ValueError('Publisher scheduler configuration changed')
    vendor = ROOT/'vendor/sdxl-train'
    for name, expected in protocol['shared_trainer']['files'].items():
        if sha(vendor/name) != expected:
            raise ValueError('Pinned shared trainer changed: '+name)
    return protocol, manifest


def validate_continuation(path, protocol):
    """Require the root's hash-bound native visual-benefit decision before fresh100."""
    if path is None:
        raise ValueError('100 updates require --continue-decision from the20-update native review')
    decision = json.loads(path.read_text())
    if decision['protocol_sha256'] != sha(PROTOCOL) or decision['continue100'] is not True:
        raise ValueError('Missing positive decision for this frozen protocol')
    for artifact in decision['artifacts']:
        file = ROOT.parents[1] / artifact['path']
        if sha(file) != artifact['sha256']:
            raise ValueError('Continuation review artifact changed: '+artifact['path'])
    kinds = {a['kind'] for a in decision['artifacts']}
    if not {'training20_evidence','adapter20','base_development_result','adapter20_development_result','native_visual_review'}.issubset(kinds):
        raise ValueError('Continuation lacks necessary evidence artifacts')
    summary = decision['summary']
    if summary['reviewed_cases'] != 24 or summary['reviewed_primary_arms'] != 2:
        raise ValueError('Must review all24 cases in both primary arms')
    if summary['photographic_wins'] <= summary['photographic_losses'] or summary['photographic_wins'] < 1:
        raise ValueError('No actual native photographic benefit')
    if summary['calendar_wins'] < summary['calendar_losses']:
        raise ValueError('Calendar appeal regressed')
    for metric in ('joint_pass','face_pass'):
        if summary['adapter_'+metric] < summary['base_'+metric]:
            raise ValueError('Native quality count regressed: '+metric)
    if summary['new_severe_face_or_identity_regression'] or summary['strong_teacher_copies']:
        raise ValueError('Face, identity or memorization regression')
    evidence_artifact = next(a for a in decision['artifacts'] if a['kind']=='training20_evidence')
    evidence = json.loads((ROOT.parents[1]/evidence_artifact['path']).read_text())
    if evidence['steps'] != 20 or not evidence['frozen_base_exactly_unchanged'] or not evidence['changed_adapter_keys']:
        raise ValueError('20-update adapter/base evidence failed')
    return decision
