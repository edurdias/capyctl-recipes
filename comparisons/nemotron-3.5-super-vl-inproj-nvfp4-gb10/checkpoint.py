"""Read tensor spans and rewrite safetensors without materializing a shard."""
import hashlib
import json
from pathlib import Path
import struct

import torch

BUFFER = 8 << 20
DTYPES = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'F32': torch.float32}


def read_header(path):
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    return header, length + 8


def digest(path, start=0, length=None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(start)
        remaining = Path(path).stat().st_size - start if length is None else length
        while remaining:
            block = stream.read(min(BUFFER, remaining))
            if not block:
                raise EOFError(path)
            result.update(block)
            remaining -= len(block)
    return result.hexdigest()


def tensor_digest(directory, meta):
    start, end = meta['data_offsets']
    return digest(Path(directory) / meta['shard'], meta['header_bytes'] + start, end - start)


def read_tensor(directory, meta):
    start, end = meta['data_offsets']
    with (Path(directory) / meta['shard']).open('rb') as stream:
        stream.seek(meta['header_bytes'] + start)
        data = bytearray(end - start)
        if stream.readinto(data) != len(data):
            raise EOFError(meta['shard'])
    return torch.frombuffer(data, dtype=DTYPES[meta['dtype']]).reshape(meta['shape'])


def headers(directory, check_total=False):
    directory = Path(directory)
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    result = {}
    for shard in sorted(set(index['weight_map'].values())):
        header, size = read_header(directory / shard)
        cursor = 0
        entries = sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                         key=lambda item: item[1]['data_offsets'])
        for name, meta in entries:
            start, end = meta['data_offsets']
            if start != cursor or end < start or name in result:
                raise ValueError('Invalid tensor spans: ' + name)
            cursor = end
            result[name] = {**meta, 'shard': shard, 'header_bytes': size}
        if size + cursor != (directory / shard).stat().st_size:
            raise ValueError('Invalid shard payload size: ' + shard)
    actual_map = {k: v['shard'] for k, v in result.items()}
    if actual_map != index['weight_map']:
        raise ValueError('Tensor index does not exactly cover physical headers')
    if check_total and index['metadata']['total_size'] != payload_size(result):
        raise ValueError('Tensor index total_size mismatch')
    return result


def payload_size(metadata):
    return sum(m['data_offsets'][1] - m['data_offsets'][0] for m in metadata.values())


def selected_bases(config):
    return [f'language_model.backbone.layers.{i}.mixer.in_proj'
            for i, kind in enumerate(config['llm_config']['layers_block_type']) if kind == 'mamba']


def verify_source(directory, inventory):
    directory = Path(directory)
    for name, expected in inventory['metadata_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError('Pinned metadata hash mismatch: ' + name)
    for name, expected in inventory['shards'].items():
        if (directory / name).stat().st_size != expected['bytes'] or digest(directory / name) != expected['sha256']:
            raise ValueError('Pinned shard hash mismatch: ' + name)


def write_shard(source, destination, replacements):
    """Copy untouched payload spans verbatim; replacement tensors stay CPU-only."""
    header, header_bytes = read_header(source)
    output = {'__metadata__': header['__metadata__']} if '__metadata__' in header else {}
    names = sorted((set(header) - {'__metadata__'}) | set(replacements))
    cursor = 0
    dtype_names = {value: key for key, value in DTYPES.items()}
    for name in names:
        if name in replacements:
            value = replacements[name]
            count = value.numel() * value.element_size()
            meta = {'dtype': dtype_names[value.dtype], 'shape': list(value.shape)}
        else:
            meta = dict(header[name])
            start, end = meta['data_offsets']
            count = end - start
        output[name] = {**meta, 'data_offsets': [cursor, cursor + count]}
        cursor += count
    encoded = json.dumps(output, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    with Path(source).open('rb') as incoming, Path(destination).open('xb') as outgoing:
        outgoing.write(struct.pack('<Q', len(encoded)))
        outgoing.write(encoded)
        for name in names:
            if name in replacements:
                # Zero-copy view; no extra tensor-sized Python bytes allocation.
                value = replacements[name].detach().contiguous().reshape(-1).view(torch.uint8)
                outgoing.write(memoryview(value.numpy()))
            else:
                start, end = header[name]['data_offsets']
                incoming.seek(header_bytes + start)
                remaining = end - start
                while remaining:
                    block = incoming.read(min(BUFFER, remaining))
                    if not block:
                        raise EOFError(name)
                    outgoing.write(block)
                    remaining -= len(block)
    return output
