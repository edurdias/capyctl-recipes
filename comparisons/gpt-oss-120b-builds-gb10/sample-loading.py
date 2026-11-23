#!/usr/bin/env python3
"""Observe an owned run's loading progress and minimum available host memory."""
import argparse
import json
from pathlib import Path
import re
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('work', type=Path)
a = p.parse_args()
cells = {}
while True:
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    logs = '\n'.join(path.read_text(errors='replace') for path in (a.work/'state/logs').rglob('*.log'))
    matches = re.findall(r'\|\s*(\d+)/(\d+)\s*\[', logs)
    key = '/'.join(matches[-1]) if matches else 'before-shard-progress'
    cell = cells.setdefault(key, {'first_unix_s': time.time(), 'samples': 0, 'min_mem_available_bytes': available})
    cell['samples'] += 1
    cell['last_unix_s'] = time.time()
    cell['min_mem_available_bytes'] = min(cell['min_mem_available_bytes'], available)
    (a.work/'loading-memory-by-shard.json').write_text(json.dumps(cells, indent=2)+'\n')
    result = a.work/'run.json'
    if result.exists():
        state = json.loads(result.read_text())
        if state.get('load_completed') or state.get('status') == 'blocked':
            break
    time.sleep(.5)
print(json.dumps(cells), flush=True)
