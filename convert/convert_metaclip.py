#!/usr/bin/env python3
"""MetaCLIP-h14 TEXT tower -> GGUF (f32). Pre-LN CLIP text transformer: token+pos emb,
24 layers (LN1->causal MHA->res, LN2->quickgelu MLP->res), final LN, text_projection."""
import argparse, torch, json
from gguf import GGUFWriter
from transformers import AutoModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/media/ilintar/D_SSD/thinksound/hf/metaclip-h14")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/gguf/metaclip-text-f32.gguf")
    a = ap.parse_args()
    tc = json.load(open(f"{a.path}/config.json"))["text_config"]
    sd = AutoModel.from_pretrained(a.path, dtype=torch.float32).eval().state_dict()
    H = tc["hidden_size"]; NL = tc["num_hidden_layers"]; NH = tc["num_attention_heads"]
    tok = sd["text_model.embeddings.token_embedding.weight"]
    pos = sd["text_model.embeddings.position_embedding.weight"]

    w = GGUFWriter(a.out, arch="thinksound-metaclip-text")
    w.add_uint32("clip.hidden", H); w.add_uint32("clip.layers", NL); w.add_uint32("clip.heads", NH)
    w.add_uint32("clip.vocab", tok.shape[0]); w.add_uint32("clip.ctx", pos.shape[0])
    w.add_uint32("clip.intermediate", tc["intermediate_size"]); w.add_float32("clip.eps", 1e-5)
    w.add_uint32("clip.eos_id", 49407); w.add_uint32("clip.bos_id", 49406)

    def put(n, t): w.add_tensor(n, t.to(torch.float32).contiguous().numpy())
    put("tok_emb", tok); put("pos_emb", pos)
    put("out_ln.w", sd["text_model.final_layer_norm.weight"]); put("out_ln.b", sd["text_model.final_layer_norm.bias"])
    put("text_proj", sd["text_projection.weight"])
    for i in range(NL):
        p = f"text_model.encoder.layers.{i}"
        for x in ("q","k","v","out"):
            put(f"blk.{i}.{x}.w", sd[f"{p}.self_attn.{x}_proj.weight"]); put(f"blk.{i}.{x}.b", sd[f"{p}.self_attn.{x}_proj.bias"])
        put(f"blk.{i}.ln1.w", sd[f"{p}.layer_norm1.weight"]); put(f"blk.{i}.ln1.b", sd[f"{p}.layer_norm1.bias"])
        put(f"blk.{i}.ln2.w", sd[f"{p}.layer_norm2.weight"]); put(f"blk.{i}.ln2.b", sd[f"{p}.layer_norm2.bias"])
        put(f"blk.{i}.fc1.w", sd[f"{p}.mlp.fc1.weight"]); put(f"blk.{i}.fc1.b", sd[f"{p}.mlp.fc1.bias"])
        put(f"blk.{i}.fc2.w", sd[f"{p}.mlp.fc2.weight"]); put(f"blk.{i}.fc2.b", sd[f"{p}.mlp.fc2.bias"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
