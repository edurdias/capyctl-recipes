#!/usr/bin/env python3
"""Print tables from retained evidence without running inference."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = (("capyctl-vl032", "CapyCTL · vLLM 0.32"),
          ("openai-vl032", "OpenAI · vLLM 0.32"),
          ("published-vl030", "Published OpenAI · vLLM 0.30"))


def markdown(matrix):
    lines = ["| Setup | Short c1 decode | c2 aggregate | c4 aggregate | c8 aggregate | Decode 8k / 32k / 64k / 110k |",
             "|---|---:|---:|---:|---:|---|"]
    for key, label in SERIES:
        row = matrix["rows"][key]
        short = [f'{cell["decode_tps" if cell["streams"] == 1 else "aggregate_tps"]:.3f}' for cell in row["short"]]
        long = " / ".join(f'{cell["decode_tps"]:.3f}' for cell in row["long_decode"])
        lines.append("| " + " | ".join([label, *short, long]) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(json.loads((HERE / "bench/final-matrix.json").read_text())))
