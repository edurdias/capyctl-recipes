#!/usr/bin/env python3
"""Audit retained publication text; private cohorts are read only for leak checks."""
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main():
    private = Path.home() / 'clef-decision-eval'
    texts = set()
    for name in ('diagnostic-104.json', 'heldout.json'):
        for row in json.loads((private / name).read_text()):
            for value in strings({'state': row['state'], 'questions': row['questions']}):
                if len(value) >= 80:
                    texts.add(value)
    paths = list(HERE.rglob('*'))
    paths.extend((REPO / 'sglang' / 'clef-flash-capyctl-mixed-gb10').rglob('*'))
    identity_hits = []
    cohort_hits = []
    inventories = []
    for path in paths:
        if not path.is_file() or path.name == 'publication-audit.json':
            continue
        data = path.read_bytes()
        try:
            text = data.decode()
        except UnicodeDecodeError:
            continue
        rel = str(path.relative_to(REPO))
        if re.search(r'/home/(?!me(?:/|[\s\"\']|$))[A-Za-z_][A-Za-z0-9_.-]*', text):
            identity_hits.append(rel)
        if any(value in text or json.dumps(value, ensure_ascii=False)[1:-1] in text for value in texts):
            cohort_hits.append(rel)
        inventories.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    caches = [str(p.relative_to(REPO)) for p in paths if p.name in ('NOTES.md', '__pycache__') or p.suffix == '.pyc']
    numeric_evidence = []
    archive = os.environ.get('CLEF_PUBLICATION_ARCHIVE')
    def numbers(value):
        if isinstance(value, dict):
            return [n for child in value.values() for n in numbers(child)]
        if isinstance(value, list):
            return [n for child in value for n in numbers(child)]
        return [value] if isinstance(value, (int, float)) else []
    if archive:
        originals = list((Path(archive) / 'comparison-original/results').glob('*'))
        originals.extend((Path(archive) / 'laptop-r1-original').glob('*'))
        for original in originals:
            current = HERE / 'results' / original.name
            if original.suffix not in ('.json', '.jsonl') or not current.is_file():
                continue
            def load(path):
                text = path.read_text()
                return [json.loads(line) for line in text.splitlines()] if path.suffix == '.jsonl' else json.loads(text)
            unchanged = numbers(load(original)) == numbers(load(current))
            numeric_evidence.append({'file': original.name, 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'all_numeric_values_preserved': unchanged})
            if not unchanged:
                identity_hits.append(str(current.relative_to(REPO)) + ': numeric evidence changed')
    audit = {
        'status': 'complete' if not (identity_hits or cohort_hits or caches) else 'blocked',
        'private_notes': 'Archived with an untouched copy of the original comparison outside the repository; no working notes or generated Python caches are shipped.',
        'private_request_text': {'scan': 'Exact state/question string leaves at least 80 characters, both literal and JSON-escaped; private cohort bytes stay outside the repository.', 'strings_scanned': len(texts), 'matching_files': cohort_hits, 'output_redaction': 'Question and categorical option labels are SHA-256 identifiers; score legends removed. Ordered-payload hashes and all probability values retained.'},
        'identity_scan': {'home_replacement': '/home/me', 'private_home_path_matching_files': identity_hits},
        'excluded_files_remaining': caches,
        'configuration_validation': {'tool': 'capyctl 0.1.12 validate config', 'mixed': 'valid deployment document; revalidated after mixed-only publication decision', 'limitations': 'No placement, runtime profile, resource-policy, route, Hub provisioning or accepted-checkpoint digest qualification.'},
        'numeric_evidence': numeric_evidence,
        'publication_decision': {'published': ['capyctl/clef-flash-NVFP4-FP8'], 'not_published': {'checkpoint': 'clef-flash-capyctl-fp8-r1', 'gb10_agreement_percent': 98.564, 'laptop_agreement_percent': 98.357, 'published_laptop_fp8_agreement_percent': 98.451, 'sampled_device_headroom_mib': 23.1875, 'reason': 'Same-laptop agreement regresses and memory headroom is extremely thin. FP8 recipe, model card and upload inventory removed; measured comparison evidence retained.'}},
        'laptop': {
            'agreement': json.loads((HERE / 'results/laptop-capyctl-fp8-r1-vs-bf16-heldout.json').read_text()),
            'run': json.loads((HERE / 'results/laptop-capyctl-fp8-r1-run.json').read_text()),
            'release': json.loads((HERE / 'results/laptop-capyctl-fp8-r1-release.json').read_text()),
        },
        'judgement_calls': [
            'I publish only the GB10 mixed build despite missing 99% held-out BF16 agreement, because its same-top agreement exceeds published NVFP4.',
            'Overlapping question-level Wilson intervals are descriptive and do not establish statistical superiority.',
            'FP8 r1 worsens mean KL/TV versus published FP8; mixed remains worse than published FP8.',
            'Ada r1 agreement is 5207/5294, lower than published laptop FP8 at 5212/5294; the GB10 rationale does not generalize to an Ada improvement.',
            'The Ada run completed but sampled only 23.1875 MiB device headroom; I do not claim safe general-purpose 16 GB deployment residency.',
            'Sensitivity/layout selection is BF16-arithmetic QDQ simulation, not quantized-kernel equivalence or engine qualification.',
            'GPU-memory peaks on GB10 are unmeasured; system MemAvailable is reported separately.',
            '1e72872e4c70b563038336637391e05704f2ccb4 is intentional until upload. Only the mixed checkpoint inventory is staged; its model card replaces upstream README metadata at upload time.',
            'Only direct-engine measurements are claimed. The deployment resource requests do not prove host placement or reproduce a fixed static memory fraction.'
        ],
        'upload_status': 'Mixed file list staged only; FP8 is not published or staged. No upload or commit performed.',
        'files': inventories,
        'unresolved': identity_hits + cohort_hits + caches,
    }
    (HERE / 'publication-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'text_files': len(inventories), 'unresolved': audit['unresolved']}))
    if audit['unresolved']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
