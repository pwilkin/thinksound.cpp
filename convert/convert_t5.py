#!/usr/bin/env python3
"""T5-v1.1-xl ENCODER -> GGUF (f32). Standard T5 encoder: shared embedding, 24 blocks
(RMSNorm, self-attn w/ shared relative-position bias in block 0, gated-gelu FFN), final RMSNorm."""
import argparse, torch, json, re, numpy as np
from gguf import GGUFWriter, GGMLQuantizationType
import gguf.quants as gq
from transformers import T5EncoderModel

KEEP_F32 = {"tok_emb", "rel_bias", "out_norm"}  # embedding, rel-bias, final norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/media/ilintar/D_SSD/thinksound/hf/t5-v1_1-xl")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/gguf/t5-f32.gguf")
    ap.add_argument("--quant", default="none", choices=["none", "q8_0"])
    a = ap.parse_args()
    cfg = json.load(open(f"{a.path}/config.json"))
    sd = T5EncoderModel.from_pretrained(a.path, dtype=torch.float32).eval().state_dict()

    w = GGUFWriter(a.out, arch="thinksound-t5")
    w.add_uint32("t5.d_model", cfg["d_model"]); w.add_uint32("t5.d_ff", cfg["d_ff"])
    w.add_uint32("t5.n_layers", cfg["num_layers"]); w.add_uint32("t5.n_heads", cfg["num_heads"])
    w.add_uint32("t5.d_kv", cfg["d_kv"]); w.add_uint32("t5.vocab", cfg["vocab_size"])
    w.add_uint32("t5.rel_buckets", cfg["relative_attention_num_buckets"])
    w.add_uint32("t5.rel_max_dist", cfg.get("relative_attention_max_distance", 128) or 128)
    w.add_float32("t5.eps", cfg["layer_norm_epsilon"])

    nq = [0]
    def put(name, t):
        arr = np.ascontiguousarray(t.to(torch.float32).numpy())
        if a.quant == "q8_0" and arr.ndim == 2 and name not in KEEP_F32 and arr.shape[-1] % 32 == 0:
            w.add_tensor(name, gq.quantize(arr, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0); nq[0] += 1
        else:
            w.add_tensor(name, arr)
    put("tok_emb", sd["shared.weight"])
    put("rel_bias", sd["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"])  # [32buckets,32heads]
    put("out_norm", sd["encoder.final_layer_norm.weight"])
    nl = cfg["num_layers"]
    for i in range(nl):
        p = f"encoder.block.{i}.layer"
        put(f"blk.{i}.attn_norm", sd[f"{p}.0.layer_norm.weight"])
        for x in ("q","k","v","o"): put(f"blk.{i}.{x}", sd[f"{p}.0.SelfAttention.{x}.weight"])
        put(f"blk.{i}.ffn_norm", sd[f"{p}.1.layer_norm.weight"])
        for x in ("wi_0","wi_1","wo"): put(f"blk.{i}.{x}", sd[f"{p}.1.DenseReluDense.{x}.weight"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out} (quant={a.quant}, {nq[0]} tensors quantized)")

if __name__ == "__main__":
    main()
