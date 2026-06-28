# ThinkSound → GGUF Conversion & Quantization

> Companion to `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md`. Source checkpoints: `thinksound_light.ckpt` (DiT), `vae.ckpt` (oobleck), plus HF `google/t5-v1_1-xl` and `facebook/metaclip-h14-fullcc2.5b`. Targets `/media/ilintar/D_SSD/thinksound/gguf/`.
>
> Items marked **[confirm M0]** are derived from code/config and must be verified by introspecting the actual checkpoint once downloaded.

## 1. Artifacts produced

| GGUF | Source | Approx f16 size | Notes |
|---|---|---|---|
| `thinksound_light-<q>.gguf` | `thinksound_light.ckpt` | ~2.8 GB | MM-DiT (inference/EMA weights only) |
| `vae-f16.gguf` | `vae.ckpt` (decoder subset) | ~0.3 GB | oobleck **decoder + bottleneck** only; encoder dropped |
| `t5-v1_1-xl-<q>.gguf` | HF t5-v1.1-xl | ~2.4 GB | via llama.cpp `convert_hf_to_gguf.py` (T5ENCODER) |
| `metaclip-h14-text-<q>.gguf` | HF metaclip-h14 | ~0.7 GB | **text tower only**; vision tower dropped |

## 2. Source checkpoint structure  **[confirm M0]**

From `predict.py`: `model = create_model_from_config(cfg); model.load_state_dict(torch.load("thinksound_light.ckpt"))`. So the light ckpt is a *bare* `state_dict` for the top-level `mm_diffusion_cond` module (no Lightning wrapper — `unwrap.py` already stripped that). Expected top-level prefixes:

- `model.model.*` — the `MMmodule` DiT (the wrapper nests the raw net under `.model.model`; **[confirm M0]** whether it is `model.*` or `model.model.*`).
- `pretransform.*` — the VAE (but in `predict.py` the VAE is loaded **separately** from `vae.ckpt` with `prefix='autoencoder.'`, so the DiT ckpt's pretransform tensors may be absent/ignored).
- `conditioner.*` — the 4 `mm_unchang` conditioners (pass-through; likely **no learned weights** — verify).

`vae.ckpt`: loaded via `load_ckpt_state_dict(path, prefix='autoencoder.')` → keys like `autoencoder.decoder.layers.*`, `autoencoder.encoder.*` (encoder dropped), `autoencoder.bottleneck.*`. Contains weight-norm pairs `*.weight_g` / `*.weight_v`.

`unwrap.py` is the authority on what the “light”/inference ckpt contains (EMA vs raw, what is stripped); read it during M0 to lock the exact key map.

### Why the sizes
DiT ≈ **1.3–1.5 B params** (hidden 1024; 7 joint blocks ×3 sub-blocks + 14 fused blocks, SwiGLU conv-FFNs dominate). `5.73 GB` ≈ that param count as fp32 model **or** (model+EMA) fp16 → we extract the single inference copy (EMA preferred — it is the weights `predict.py` uses). The full `21 GB` ckpt additionally carries AdamW optimizer state (2 moments) — irrelevant to us.

## 3. Weight-norm folding (VAE convs)

Every `WNConv1d`/`WNConvTranspose1d` stores `weight_g` (shape `[OC,1,1]`) + `weight_v` (`[OC,IC,K]`), `weight = g · v / ‖v‖` with the norm over dims `(IC,K)` per output channel (torch `weight_norm` default `dim=0`). The converter folds to a single plain `weight`:

```python
import torch
def fold_weight_norm(g, v):                 # g:[OC,1,1]  v:[OC,IC,K]
    norm = v.flatten(1).norm(dim=1).view(-1, 1, 1)
    return g * v / norm
```

ConvTranspose weight-norm uses `dim=0` too but torch lays out transpose weights as `[IC,OC,K]`; **[confirm M0]** the norm dim for `WNConvTranspose1d` (dac uses `dim=1` for some) and transpose the kernel to ggml's expected `conv_transpose_1d` layout at convert time.

## 4. Tensor naming convention

Keep names close to the torch state_dict (readable, greppable), lowercased, dots preserved. Examples:

```
# DiT
dit.audio_input_proj.0.weight            # ChannelLastConv1d k3
dit.t_embed.mlp.0.weight
dit.joint_blocks.0.latent_block.attn.qkv.weight
dit.joint_blocks.0.latent_block.attn.q_norm.weight
dit.joint_blocks.0.latent_block.adaLN_modulation.1.weight
dit.joint_blocks.0.text_block.ffn.w1.weight
dit.fused_blocks.5.linear1.weight
dit.final_layer.conv.weight
dit.empty_clip_feat   dit.empty_sync_feat   dit.empty_string_feat   dit.empty_t5_feat
dit.sync_pos_emb
# VAE
vae.decoder.layers.0.weight              # initial conv (post weight-norm fold)
vae.decoder.layers.1.layers.1.weight     # DecoderBlock upsample (transpose conv)
vae.decoder.layers.1.layers.2.layers.1.alpha   # SnakeBeta alpha (residual unit)
```
A small `name_map` table in each converter makes the C++ loader’s lookups exact. The C++ side reads names verbatim — no implicit remapping.

## 5. GGUF metadata (hparams)

Embed under a `thinksound.*` / `vae.*` namespace so the C++ loader needs no external config:

```
general.architecture = "thinksound-mmdit"
thinksound.latent_dim=64  thinksound.hidden_dim=1024  thinksound.depth=21
thinksound.fused_depth=14  thinksound.num_heads=16  thinksound.clip_dim=1024
thinksound.sync_dim=768   thinksound.text_dim=2048   thinksound.kernel_size=3
thinksound.v2=true  thinksound.diffusion_objective="rectified_flow"
thinksound.sample_rate=44100  thinksound.audio_channels=2
thinksound.rope_theta=10000   thinksound.t_embed_max_period=1
# seq lengths are duration-dependent → computed at runtime, not baked
vae.channels=128  vae.c_mults=[1,2,4,8,16]  vae.strides=[2,4,4,8,8]
vae.latent_dim=64  vae.use_snake=true  vae.final_tanh=false  vae.downsampling_ratio=2048
```

## 6. Quantization strategy

Estimated runtime footprint:

| Component | params | f16 | Q8_0 | Q4_K |
|---|---|---|---|---|
| MM-DiT | ~1.4 B | ~2.8 GB | ~1.5 GB | ~0.85 GB |
| T5-xl encoder | ~1.2 B | ~2.4 GB | ~1.3 GB | ~0.7 GB |
| MetaCLIP text | ~0.36 B | ~0.7 GB | ~0.4 GB | — |
| VAE decoder | ~0.1 B | ~0.3 GB | (keep f16) | (keep f16) |
| **Total** | | **~6.2 GB** | **~3.5 GB** | **~2.6 GB** |

Fits the RTX 5060 Ti (16 GB) at f16 with room; Q8 fits the RTX 3080 (10 GB) comfortably.

**Quantize (per-tensor, row-wise along `ne0`):**
- DiT attention `qkv`/`to_q`/`to_kv`, adaLN linears, `linear1` conv, SwiGLU `w1/w2/w3` (Linear and ConvMLP). Conv weights are quantizable since inference runs them as `im2col`→matmul (the `[OC, IC·K]` matrix quantizes cleanly).
- T5 encoder linears (standard llama.cpp Q8_0/Q6_K/Q4_K path).
- MetaCLIP text linears.

**Keep at f16/f32 (never quantize):**
- All norms (RMSNorm `q_norm/k_norm` scales; LayerNorms are affine-free here), all biases.
- SnakeBeta `alpha`/`beta`, VAE convs (small + quality-critical → whole VAE decoder f16).
- Learned conditioning params: `empty_clip_feat`, `empty_sync_feat`, `empty_string_feat`, `empty_t5_feat`, `sync_pos_emb`.
- DiT `final_layer.conv` (output projection — quality-sensitive, tiny).
- Token/positional embeddings (T5 relative-bias, CLIP token+pos).

**Recommended default:** `Q8_0` for DiT + T5 (near-lossless for diffusion velocity fields), f16 VAE + CLIP. Offer `Q6_K`/`Q4_K` DiT variants and measure drift in M7 with: (a) audio aesthetics score, (b) log-spectral distance vs the f16 reference, (c) listening. Diffusion models tolerate weight quant well but **velocity-field error compounds over 24 steps** — validate, don’t assume.

## 7. Converter implementation notes

- Use `gguf` Python lib (already in `/devel/tools/llama.cpp/gguf-py`).
- `torch.load(..., map_location="cpu", weights_only=True)`; iterate state_dict; fold weight-norm; cast (f32 kept tensors / quantize target tensors written as f16 then quantized by a `quantize` pass or directly via gguf quant writers).
- Deterministic tensor ordering; assert the full expected set is present (fail loud on missing/extra keys) — this is the M0 acceptance gate.
- For T5: prefer the upstream `convert_hf_to_gguf.py --model-name t5` path to inherit its tokenizer/vocab handling; verify it emits a `t5encoder` GGUF.
- Keep a `--dtype {f16,q8_0,q6_k,q4_k}` flag per converter; emit one GGUF per requested dtype.
