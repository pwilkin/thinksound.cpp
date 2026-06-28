#!/usr/bin/env python3
"""Convert the ThinkSound MM-DiT (thinksound_light.ckpt -> model.model.*) to GGUF.
No weight-norm in the DiT; just strip the `model.model.` prefix and write f32.
Buffers t_embed.freqs / latent_rot / clip_rot are recomputed in C++, skipped here.
"""
import argparse, torch, json
from gguf import GGUFWriter, GGMLQuantizationType
import gguf.quants as gq

SKIP_SUFFIX = ("t_embed.freqs", "latent_rot", "clip_rot")

def maybe_quant(w, name, arr, quant):
    # quantize 2D linear weights (qkv/adaLN/proj/text-MLP) to Q8_0; keep convs(3D),
    # norms/biases(1D), embeddings & learned empties (not ".weight") as f32.
    if quant == "q8_0" and arr.ndim == 2 and name.endswith(".weight") and arr.shape[-1] % 32 == 0:
        import numpy as np
        qd = gq.quantize(np.ascontiguousarray(arr), GGMLQuantizationType.Q8_0)
        w.add_tensor(name, qd, raw_dtype=GGMLQuantizationType.Q8_0)
        return True
    w.add_tensor(name, arr)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/thinksound_light.ckpt")
    ap.add_argument("--cfg",  default="/tmp/ThinkSound/ThinkSound/configs/model_configs/thinksound.json")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/gguf/dit-f32.gguf")
    ap.add_argument("--quant", default="none", choices=["none", "q8_0"])
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    dit = {k[len("model.model."):]: v for k, v in sd.items() if k.startswith("model.model.")}
    dit = {k: v for k, v in dit.items() if not k.endswith(SKIP_SUFFIX)}
    print(f"DiT tensors: {len(dit)}")

    cfg = json.load(open(args.cfg))["model"]["diffusion"]["config"]
    w = GGUFWriter(args.out, arch="thinksound-mmdit")
    for k, v in dict(
        latent_dim=64, clip_dim=1024, sync_dim=768, text_dim=2048, hidden_dim=1024,
        depth=21, fused_depth=14, num_heads=16, text_seq_len=77, kernel_size=3, sync_kernel=7,
    ).items():
        w.add_uint32(f"dit.{k}", int(cfg.get(k, v)))
    w.add_float32("dit.rope_theta", 10000.0)
    w.add_float32("dit.t_embed_max_period", 1.0)
    w.add_uint32("dit.sample_rate", 44100)
    w.add_string("dit.diffusion_objective", "rectified_flow")

    n = 0; nq = 0
    for name, t in dit.items():
        arr = t.to(torch.float32).contiguous().numpy()
        if maybe_quant(w, name, arr, args.quant): nq += 1
        n += t.numel()
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out} ({n/1e9:.3f}B params, quant={args.quant}, {nq} tensors quantized)")

if __name__ == "__main__":
    main()
