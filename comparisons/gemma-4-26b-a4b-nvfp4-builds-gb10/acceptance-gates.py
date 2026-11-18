#!/usr/bin/env python3
"""Measure live retrieval, reasoning separation, tools, and vision acceptance.

Use the existing engine Python environment. This script never launches an engine
or samples memory. The first HTTP request is the cold retrieval probe. Results
are saved after every completed probe, including each concurrent soak request.
Transport secrets, addresses, and serving identity metadata are redacted; prompt
and answer evidence, including fresh random codes, is otherwise retained.
"""

import argparse
import base64
import concurrent.futures
import importlib.util
import io
import json
from pathlib import Path
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


TARGET_TOKENS = 111_000
CODE_PATTERN = r"\bGATE[0-9A-F]{16}\b"
DEPTHS = (0.15, 0.50, 0.85)


def fresh_code():
    return "GATE" + secrets.token_hex(8).upper()


def load_benchmark():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "capyctl_bench", root / "tools/capyctl-bench/capyctl_bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def token_count(tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def prepare_retrieval(tokenizer, benchmark, probe_id, seed):
    codes = []
    while len(codes) < 3:
        code = fresh_code()
        if code not in codes:
            codes.append(code)
    # Generate one varied pseudo-English document, then size it by actual chat
    # tokens. Offset mapping keeps inserted records inside the original text.
    filler = benchmark.filler_text(16_000, seed)
    offsets = tokenizer(
        filler, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    prefix = (
        f"Document {seed}. Read this document and find its three retrieval records.\n"
        "Return their codes in document order, exactly as written.\n\n"
        "BEGIN DOCUMENT\n"
    )
    suffix = (
        "\nEND DOCUMENT\n\n"
        "Request: return only the three retrieval record codes in document order."
    )

    def assemble(size, padding=""):
        end = offsets[size - 1][1] if size else 0
        cursor = 0
        parts = []
        for depth, code in zip(DEPTHS, codes):
            position = offsets[int(size * depth)][0] if size else 0
            parts.extend((filler[cursor:position], f"\nRetrieval record: {code}\n"))
            cursor = position
        parts.extend((filler[cursor:end], padding))
        document = "".join(parts)
        messages = [{"role": "user", "content": prefix + document + suffix}]
        return messages, document

    low, high = 0, min(len(offsets), TARGET_TOKENS + 4096)
    if token_count(tokenizer, assemble(high)[0]) <= TARGET_TOKENS:
        raise ValueError("Generated filler is too short for the target")
    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(tokenizer, assemble(middle)[0]) <= TARGET_TOKENS:
            low = middle
        else:
            high = middle
    messages, document = assemble(low)
    count = token_count(tokenizer, messages)
    padding = ""
    # Usually only a token or two remain. Use distinct ordinary words rather
    # than the repetitive single-word padding used by shorter context fixtures.
    used = set()
    while count < TARGET_TOKENS:
        for word in benchmark.WORDS:
            if word in used:
                continue
            candidate, candidate_document = assemble(low, padding + " " + word)
            candidate_count = token_count(tokenizer, candidate)
            if count < candidate_count <= TARGET_TOKENS:
                used.add(word)
                padding += " " + word
                messages, document, count = candidate, candidate_document, candidate_count
                break
        else:
            raise ValueError("Exact template token padding could not be constructed")
    document_tokens = len(tokenizer.encode(document, add_special_tokens=False))
    insertions = []
    for depth, code in zip(DEPTHS, codes):
        character_offset = document.index(code)
        token_offset = len(tokenizer.encode(document[:character_offset], add_special_tokens=False))
        insertions.append({
            "code": code,
            "target_depth_fraction": depth,
            "document_character_offset": character_offset,
            "document_token_offset": token_offset,
            "actual_depth_fraction": token_offset / document_tokens,
        })
    return {
        "probe_id": probe_id,
        "filler_seed": seed,
        "expected_codes": codes,
        "local_prompt_tokens": count,
        "document_tokens": document_tokens,
        "insertions": insertions,
        "messages": messages,
    }


class Client:
    def __init__(self, endpoint, model, key):
        parsed = urllib.parse.urlsplit(endpoint.rstrip("/"))
        path = parsed.path.rstrip("/")
        if not path.endswith("/v1"):
            path += "/v1"
        self.url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/chat/completions", "", ""))
        self.model = model
        self.key = key
        self.private_strings = sorted(
            {value for value in (key, endpoint, parsed.netloc, parsed.hostname, model) if value},
            key=len, reverse=True,
        )

    def public(self, value):
        """Keep evidence, but not transport/serving identity metadata."""
        if isinstance(value, dict):
            return {
                name: ("[redacted serving metadata]" if name in {
                    "model", "system_fingerprint", "service_tier", "served_model_name",
                    "server", "headers", "provider", "backend", "engine", "deployment",
                    "owned_by", "organization", "request_id", "trace_id",
                } or (name == "id" and "choices" in value) else self.public(item))
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [self.public(item) for item in value]
        if isinstance(value, str):
            for private in self.private_strings:
                value = value.replace(private, "[redacted]")
            value = re.sub(r"https?://[^\s\"<>]+", "[redacted address]", value)
            value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[redacted address]", value)
            return value
        return value

    def request(self, body):
        body = {"model": self.model, "temperature": 0, "stream": False, **body}
        record = {"request": body, "response": None, "http_status": None, "error": None}
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                self.url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", **(
                    {"Authorization": "Bearer " + self.key} if self.key else {}
                )},
            )
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    record["http_status"] = response.status
                    raw = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as error:
                record["http_status"] = error.code
                raw = error.read().decode("utf-8", errors="replace")
                record["error"] = "HTTP request failed"
            try:
                record["response"] = json.loads(raw)
            except (ValueError, TypeError):
                record["response"] = {"unparsed_body": raw}
                record["error"] = record["error"] or "Response was not JSON"
        except Exception as error:
            # Exception strings can contain URLs, headers, and credential paths.
            record["error"] = "Request failed: " + type(error).__name__
        record["latency_seconds"] = time.perf_counter() - started
        return record


def assistant_message(record):
    response = record.get("response")
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def content_text(message):
    content = message.get("content")
    return content if isinstance(content, str) else ""


def request_ok(record):
    status = record.get("http_status")
    return isinstance(status, int) and 200 <= status < 300 and record.get("error") is None


def retrieval_probe(client, prepared):
    record = client.request({
        "messages": prepared["messages"], "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    response = record.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    api_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    extracted = re.findall(CODE_PATTERN, content_text(assistant_message(record)))
    conditions = {
        "http_succeeded": request_ok(record),
        "local_prompt_is_111000_tokens": prepared["local_prompt_tokens"] == TARGET_TOKENS,
        "api_prompt_is_approximately_111000_tokens": (
            isinstance(api_tokens, int) and 110_000 <= api_tokens <= 112_000
        ),
        "exact_three_codes_in_order": extracted == prepared["expected_codes"],
    }
    return {
        **prepared, "exchange": record, "api_prompt_tokens": api_tokens,
        "api_matches_local_token_count": api_tokens == prepared["local_prompt_tokens"],
        "extracted_codes": extracted, "pass_conditions": conditions,
        "passed": all(conditions.values()),
    }


def reasoning_probe(client):
    prompt = (
        "Find the least positive integer n such that n leaves remainder 2 when divided by 5, "
        "remainder 3 when divided by 7, and remainder 4 when divided by 9. "
        "Think through the problem, but put only the integer in your final answer."
    )
    record = client.request({
        "messages": [{"role": "user", "content": prompt}], "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": True},
    })
    message = assistant_message(record)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    content = content_text(message)
    markup = re.search(r"</?(?:think|analysis|reasoning)>|\[/?THINK\]|<\|(?:channel|im_start|im_end)[^>]*>", content, re.I)
    match = re.fullmatch(r"\s*([0-9]+)\s*", content)
    conditions = {
        "http_succeeded": request_ok(record),
        "separate_nonempty_reasoning_field": isinstance(reasoning, str) and bool(reasoning.strip()),
        "final_answer_is_only_correct_integer": bool(match and int(match.group(1)) == 157),
        "no_reasoning_markup_in_content": markup is None,
    }
    return {
        "exchange": record, "expected_answer": 157,
        "parsed_answer": int(match.group(1)) if match else None,
        "pass_conditions": conditions, "passed": all(conditions.values()),
    }


def tool_probe(client):
    code = fresh_code()
    name = "read_acceptance_code"
    tools = [{"type": "function", "function": {
        "name": name, "description": "Read the acceptance code.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }}]
    messages = [{"role": "user", "content": (
        "Call read_acceptance_code exactly once. After receiving its result, "
        "quote the returned code exactly in your final answer."
    )}]
    first = client.request({
        "messages": messages, "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": name}},
        "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
    })
    assistant = assistant_message(first)
    calls = assistant.get("tool_calls")
    valid = False
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        call = calls[0]
        function = call.get("function")
        if isinstance(function, dict):
            try:
                arguments = json.loads(function.get("arguments", ""))
            except (ValueError, TypeError):
                arguments = None
            valid = (
                assistant.get("role") == "assistant" and call.get("type") == "function"
                and isinstance(call.get("id"), str) and bool(call["id"])
                and function.get("name") == name and arguments == {}
            )
    second = None
    tool_message = None
    if request_ok(first) and valid:
        tool_message = {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"code": code})}
        # Retain the actual assistant tool_calls message, not a reconstructed call.
        second = client.request({
            "messages": messages + [assistant, tool_message], "tools": tools,
            "tool_choice": "none", "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        })
    final_message = assistant_message(second) if second else {}
    extracted = re.findall(CODE_PATTERN, content_text(final_message))
    conditions = {
        "initial_http_succeeded": request_ok(first),
        "exactly_one_valid_requested_tool_call": valid,
        "tool_result_http_succeeded": second is not None and request_ok(second),
        "answer_quotes_exact_returned_code": extracted == [code],
        "no_further_tool_calls": second is not None and not final_message.get("tool_calls"),
    }
    return {
        "expected_code": code, "initial_exchange": first,
        "assistant_tool_calls_message": assistant, "tool_result_message": tool_message,
        "answer_exchange": second, "extracted_codes": extracted,
        "pass_conditions": conditions, "passed": all(conditions.values()),
    }


def vision_probe(client, out):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (300, 176), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 40, 120, 136), fill="red")
    draw.ellipse((180, 40, 276, 136), fill="blue")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    image_path = out.with_name(out.stem + "-vision.png")
    image_path.write_bytes(encoded.getvalue())
    data_url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
    record = client.request({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": (
                "Identify the two objects in the image by their color, shape, and position. "
                "Use exactly two lines: Left: COLOR SHAPE and Right: COLOR SHAPE."
            )},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
    })
    answer = content_text(assistant_message(record))
    descriptions = {}
    for position in ("left", "right"):
        match = re.search(r"^\s*" + position + r"\s*:\s*([^\n]+)", answer, re.I | re.M)
        descriptions[position] = match.group(1).lower() if match else ""
    left_words = set(re.findall(r"[a-z]+", descriptions["left"]))
    right_words = set(re.findall(r"[a-z]+", descriptions["right"]))
    conditions = {
        "http_succeeded": request_ok(record),
        "left_is_red_square": {"red", "square"} <= left_words and not {"blue", "circle"} & left_words,
        "right_is_blue_circle": {"blue", "circle"} <= right_words and not {"red", "square"} & right_words,
    }
    return {
        "exchange": record, "image_artifact": image_path.name,
        "expected_objects": {"left": "red square", "right": "blue circle"},
        "parsed_descriptions": descriptions, "pass_conditions": conditions,
        "passed": all(conditions.values()),
    }


def failed_probe(error):
    return {
        "passed": False, "error": "Probe failed: " + type(error).__name__,
        "pass_conditions": {"probe_completed": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Root URL or URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--soak", action="store_true", help="Also run four distinct 111k-token retrieval requests concurrently")
    args = parser.parse_args()
    key = load_benchmark().read_api_key(str(args.api_key_file) if args.api_key_file else None)
    client = Client(args.endpoint, args.model, key)
    result = {
        "target_prompt_tokens": TARGET_TOKENS,
        "soak_requested": args.soak,
        "cold_request_definition": "First HTTP request; unique document; no preflight or cache reset",
        "privacy": "Serving identity metadata, secrets, and addresses are redacted; no HTTP headers are saved",
        "gates": {}, "completed": False, "overall_pass": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(client.public(result), indent=2, ensure_ascii=False) + "\n")

    def run_gate(name, operation):
        try:
            result["gates"][name] = operation()
        except Exception as error:
            result["gates"][name] = failed_probe(error)
        save()

    save()
    tokenizer = None
    benchmark = None
    seeds = secrets.SystemRandom().sample(range(1, 2**31), 5 if args.soak else 1)

    def cold():
        nonlocal tokenizer, benchmark
        from transformers import AutoTokenizer

        benchmark = load_benchmark()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        return retrieval_probe(client, prepare_retrieval(tokenizer, benchmark, "cold", seeds[0]))

    run_gate("cold_retrieval", cold)
    if args.soak:
        soak = {
            "concurrency": 4, "target_prompt_tokens": TARGET_TOKENS,
            "probes": {}, "completed": False, "passed": False,
        }
        result["gates"]["soak_retrieval"] = soak
        prepared = []
        for index in range(4):
            probe_id = f"soak-{index + 1}"
            try:
                if tokenizer is None or benchmark is None:
                    raise RuntimeError("Local tokenizer preparation was unavailable")
                prepared.append(prepare_retrieval(tokenizer, benchmark, probe_id, seeds[index + 1]))
            except Exception as error:
                soak["probes"][probe_id] = failed_probe(error)
                save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(retrieval_probe, client, item): item for item in prepared}
            for future in concurrent.futures.as_completed(pending):
                item = pending[future]
                try:
                    soak["probes"][item["probe_id"]] = future.result()
                except Exception as error:
                    soak["probes"][item["probe_id"]] = {**item, **failed_probe(error)}
                save()
        soak["completed"] = True
        seeds = [item["filler_seed"] for item in prepared]
        soak["pass_conditions"] = {
            "four_distinct_documents": len(seeds) == 4 and len(set(seeds)) == 4,
            "four_probes_passed": len(soak["probes"]) == 4 and all(probe["passed"] for probe in soak["probes"].values()),
        }
        soak["passed"] = all(soak["pass_conditions"].values())
        save()
    run_gate("reasoning_separation", lambda: reasoning_probe(client))
    run_gate("tool_round_trip", lambda: tool_probe(client))
    run_gate("vision", lambda: vision_probe(client, args.out))
    result["completed"] = True
    result["overall_pass"] = all(gate["passed"] for gate in result["gates"].values())
    save()
    print(json.dumps({
        "completed": True, "overall_pass": result["overall_pass"],
        "gates": {name: gate["passed"] for name, gate in result["gates"].items()},
    }))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
