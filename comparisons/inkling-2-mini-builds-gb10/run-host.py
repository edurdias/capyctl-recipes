#!/usr/bin/env python3
"""Host helpers adapted from the comparison precedent; never stop foreign work."""
from pathlib import Path
import subprocess


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [int(p) for p in text.split() if p.isdigit()]


def memory_fields():
    return {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:'))}


def memory():
    fields = memory_fields()
    return fields['MemTotal']-fields['MemAvailable']
