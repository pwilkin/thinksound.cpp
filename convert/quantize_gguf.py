#!/usr/bin/env python3
"""Quantize an existing f32 GGUF -> Q8_0 (linear weights only). Works from the f32
GGUF (no source checkpoint needed). Keeps convs(3D), norms/biases(1D), embeddings,
learned empties, and a name blocklist as f32."""
import argparse, numpy as np, gguf, ml_dtypes
from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType
import gguf.quants as gq

BLOCK = {"tok_emb", "rel_bias", "out_norm", "pos_emb", "text_proj"}  # never quantize

def should_q(name, ne, qtype):
    if name in BLOCK or "empty" in name or "_emb" in name or "norm" in name: return False
    if qtype == "bf16":
        return len(ne) in (2, 3)                       # linears + convs (per-element cast)
    return len(ne) == 2 and ne[0] % 32 == 0            # Q8_0: 2D linears, block-aligned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--type", default="q8_0", choices=["q8_0", "bf16"])
    a = ap.parse_args()
    r = gguf.GGUFReader(a.inp)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.out, arch=arch)
    # copy KV (skip internal + architecture which the writer adds)
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture": continue
        v = f.contents(); vt = f.types[0]
        if vt == GGUFValueType.UINT32:    w.add_uint32(k, int(v))
        elif vt == GGUFValueType.INT32:   w.add_int32(k, int(v))
        elif vt == GGUFValueType.FLOAT32: w.add_float32(k, float(v))
        elif vt == GGUFValueType.BOOL:    w.add_bool(k, bool(v))
        elif vt == GGUFValueType.STRING:  w.add_string(k, str(v))
        elif vt == GGUFValueType.ARRAY:   w.add_array(k, list(v))
    nq = 0; nf = 0
    for t in r.tensors:
        ne = [int(x) for x in t.shape]                 # ggml ne order
        arr = np.array(t.data, dtype=np.float32).reshape(list(reversed(ne)))  # numpy [.., ne0]
        if should_q(t.name, ne, a.type):
            if a.type == "q8_0":
                w.add_tensor(t.name, gq.quantize(np.ascontiguousarray(arr), GGMLQuantizationType.Q8_0),
                             raw_dtype=GGMLQuantizationType.Q8_0)
            else:  # bf16
                bf = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16)).view(np.uint8)
                w.add_tensor(t.name, bf, raw_dtype=GGMLQuantizationType.BF16)
            nq += 1
        else:
            w.add_tensor(t.name, arr); nf += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: {nq} -> {a.type}, {nf} kept f32")

if __name__ == "__main__":
    main()
