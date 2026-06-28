#!/usr/bin/env python3
"""Convert ThinkSound oobleck VAE *decoder* (vae.ckpt -> autoencoder.*) to GGUF.

Folds PyTorch weight_norm (weight_g/weight_v, dim=0) into a single `weight`.
Keeps everything f32 (VAE is small + quality-critical). Tensor names are the
torch decoder keys with the `decoder.` prefix kept (e.g. `decoder.layers.0.weight`).
"""
import argparse, re, torch, numpy as np
from gguf import GGUFWriter

# The 5 DecoderBlock upsample convs are ConvTranspose1d at decoder.layers.{1..5}.layers.1
CT_RE = re.compile(r"^decoder\.layers\.[1-5]\.layers\.1\.weight$")

def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # torch weight_norm dim=0: norm over all dims except 0, per index-0 slice.
    norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape([-1] + [1] * (v.dim() - 1))
    return g * v / norm

def to_conv1d_kernel(w: torch.Tensor) -> torch.Tensor:
    # A strided ConvTranspose1d equals: zero-stuff the input by `stride`, then a regular
    # stride-1 Conv1d with the kernel flipped along K and input/output channels swapped.
    # torch ConvTranspose1d weight [IC, OC, K] -> conv1d kernel numpy [OC, IC, K]
    # (== ggml ne=[K, IC, OC], the layout the runtime's conv1d helper expects). This runs as
    # pad + im2col + GEMM on CUDA (fast), avoiding both col2im (CPU-only) and the naive
    # CUDA conv_transpose_1d kernel (O(out*IC*OC*T_in) -- unusable for the VAE's x2048 upsample).
    return w.flip(-1).permute(1, 0, 2).contiguous()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/vae.ckpt")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/gguf/vae-f32.gguf")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    # decoder (online autoencoder, matches predict.py prefix='autoencoder.')
    dec = {k[len("autoencoder."):]: v for k, v in sd.items()
           if k.startswith("autoencoder.decoder.")}
    print(f"decoder raw keys: {len(dec)}")

    out = {}
    ct_names = []
    # fold weight_norm pairs; pass through bias/alpha/beta and any plain weights
    handled = set()
    for k in list(dec.keys()):
        if k.endswith(".weight_v"):
            base = k[:-len(".weight_v")]
            g = dec[base + ".weight_g"]; v = dec[k]
            folded = fold_weight_norm(g, v).contiguous()
            name = base + ".weight"
            if CT_RE.match(name):
                folded = to_conv1d_kernel(folded)     # [IC,OC,K] -> conv1d kernel ggml ne=[K,IC,OC]
                ct_names.append(name)
            out[name] = folded
            handled.add(base + ".weight_g"); handled.add(k)
    for k, t in dec.items():
        if k in handled: continue
        if k.endswith(".weight_g"): continue
        out[k] = t.contiguous()

    print(f"output tensors: {len(out)}; conv-transpose -> conv1d kernels [K,IC,OC]: {ct_names}")

    w = GGUFWriter(args.out, arch="thinksound-vae")
    # hparams (from thinksound.json pretransform config)
    w.add_uint32("vae.latent_dim", 64)
    w.add_uint32("vae.channels", 128)
    w.add_uint32("vae.out_channels", 2)
    w.add_array("vae.c_mults", [1, 2, 4, 8, 16])
    w.add_array("vae.strides", [2, 4, 4, 8, 8])
    w.add_uint32("vae.downsampling_ratio", 2048)
    w.add_uint32("vae.sample_rate", 44100)
    w.add_bool("vae.use_snake", True)
    w.add_bool("vae.final_tanh", False)

    for name, t in out.items():
        arr = t.to(torch.float32).numpy()
        w.add_tensor(name, arr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    total = sum(t.numel() for t in out.values())
    print(f"wrote {args.out}  ({total/1e6:.1f}M params, f32)")

if __name__ == "__main__":
    main()
