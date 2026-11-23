#!/usr/bin/env python3
"""Freeze at most three checkpoints from diagnostic-only simulations, never holdout."""
import json
from pathlib import Path
import subprocess
import sys

H = Path.home()
ROOT = H / 'clef-build'
sys.path.insert(0, str(ROOT))
from run_sensitivity import control, compare, harness
from run_candidates import engine


def main():
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not active
    subprocess.run([str(H/'modelopt-0.47.0-venv/bin/python'), str(ROOT/'tune_fp8.py')], check=True)
    units = json.loads((ROOT/'units.json').read_text())
    mapping = json.loads((ROOT/'mapping.json').read_text())
    # Exact parameter-count accounting; scalar/header differences are audited on export.
    from safetensors import safe_open
    index = json.loads((H/'models/clef-flash/model.safetensors.index.json').read_text())['weight_map']
    handles = {f:safe_open(H/'models/clef-flash'/f, framework='pt') for f in set(index.values())}
    costs = {u:0 for u in units}
    for fused, names in mapping['modules'].items():
        layer = int(fused.split('.layers.')[1].split('.')[0])
        family = 'mlp' if '.mlp.' in fused else 'attention' if '.self_attn.' in fused else 'deltanet'
        unit = f'{layer:02d}-{family}'
        for n in names:
            shape = handles[index[n+'.weight']].get_slice(n+'.weight').get_shape()
            costs[unit] += shape[0]*shape[1]
    control(mode='bf16', calibrate=False)
    measured = []
    with engine('capyctl-layout-selection', H/'models/clef-flash', hooks=True) as ep:
        def measure(label, layout, activation, tuned):
            runtime = {p:('fp8t' if m == 'fp8' and tuned else m) for u,m in layout.items() for p in units[u]}
            control(layout=runtime, activation=activation, calibrate=False)
            path = ROOT/f'{label}-diagnostic.jsonl'
            harness.capture(ep, 'diagnostic-104.json', path)
            result = compare(ROOT/'bf16-diagnostic.jsonl', path)
            rec = {'label':label, 'layout':layout.copy(), 'activation':activation, 'tuned':tuned, 'comparison':result}
            measured.append(rec)
            (ROOT/'layout-selection.json').write_text(json.dumps({'method':'BF16-arithmetic simulation only; selection uses diagnostic, not heldout', 'measurements':measured}, indent=2))
            print(label, json.dumps(result), flush=True)
            return rec
        all_fp8 = {u:'fp8' for u in units}
        for tuned in (False, True):
            for act in ('dynamic', 'static'):
                measure(f'fp8-{tuned}-{act}', all_fp8, act, tuned)
        rank = lambda r:(len(r['comparison']['flips']), r['comparison']['mean_kl'], r['comparison']['mean_tv'])
        best_fp8 = min(measured, key=rank)
        insensitive = []
        for unit in units:
            path = ROOT/f'nvfp4-{unit}-nvfp4-comparison.json'
            if path.exists():
                r = json.loads(path.read_text())
                if not r['flips']:
                    insensitive.append((r['mean_kl']/(costs[unit]*.4375), unit))
        insensitive.sort()
        for savings in (1_900_000_000, 1_700_000_000, 1_500_000_000):
            layout = all_fp8.copy(); saved = 0
            for _,unit in insensitive:
                if saved >= savings: break
                layout[unit] = 'nvfp4'; saved += costs[unit]*.4375
            measure(f'mixed-saving-{savings}', layout, best_fp8['activation'], best_fp8['tuned'])
        mixed = min((r for r in measured if r['label'].startswith('mixed-')), key=rank)
        # The optional BF16-restoration checkpoint has at most 256 MiB extra weights.
        # This is a size screen, not an Ada capacity qualification.
        fp8_rank = sorted(units, key=lambda u: json.loads((ROOT/f'fp8-{u}-comparison.json').read_text())['mean_kl'], reverse=True)
        restored = all_fp8.copy(); extra = 0
        for unit in fp8_rank:
            if extra + costs[unit] <= 256*1024**2:
                restored[unit] = 'bf16'; extra += costs[unit]
        hybrid = measure('fp8-bf16-restored', restored, best_fp8['activation'], best_fp8['tuned'])
    candidates = [dict(tag='clef-flash-capyctl-fp8-r1', layout=best_fp8['layout'], activation=best_fp8['activation'], tuned=best_fp8['tuned']), dict(tag='clef-flash-capyctl-mixed-r1', layout=mixed['layout'], activation=mixed['activation'], tuned=mixed['tuned'])]
    # Do not export the optional BF16 checkpoint without measured Ada residency.
    # Its diagnostic simulation remains useful evidence, but is not a fit claim.
    (ROOT/'candidate-spec.json').write_text(json.dumps({'selection':'diagnostic BF16-arithmetic simulations; heldout excluded; optional BF16 restoration size-screened only, Ada residency unmeasured', 'candidates':candidates}, indent=2))
    print('CANDIDATE_LAYOUTS_FROZEN', flush=True)


if __name__ == '__main__':
    main()
