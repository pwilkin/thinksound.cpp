# ThinkSound GGML Runtime — Implementation Plan

> Companion to `ARCHITECTURE.md` (component-level spec) and `GGUF_CONVERSION.md` (weight conversion + quantization).
> Scope decided 2026-06-16: **text → sound-effects only**, `thinksound_light.ckpt` (5.73 GB) + `vae.ckpt` (2.52 GB), weights/GGUF on `/media/ilintar/D_SSD/thinksound`.

## 1. Goal

A standalone C++17 binary, `thinksound-cli`, that takes a text prompt (a short *caption* + a longer *CoT description*) and emits a 44.1 kHz stereo `.wav` sound effect, running entirely on GGML (CPU + CUDA backends) with GGUF weights — no Python, no PyTorch at inference time.

```
thinksound-cli \
  --model  thinksound_light-f16.gguf \
  --vae    vae-f16.gguf \
  --t5     t5-v1_1-xl-f16.gguf \
  --clip   metaclip-h14-text-f16.gguf \
  --caption "dog barking" \
  --cot     "A medium-sized dog barks twice, sharp and close, in a quiet room." \
  --duration 9 --steps 24 --cfg 5 --seed 42 \
  -o out.wav
```

## 2. Pipeline (text path)

```
caption ─► MetaCLIP-text ─► global(1024) ──┐  per-token(77×1024)─┐
                                            │                     │
cot     ─► T5-v1.1-xl   ─► (77×2048) ───────┼─────────────────────┼─► MM-DiT
                                            │                     │   (7 joint + 14 fused,
[no video] ─► empty_clip_feat / empty_sync_feat (learned) ───────┘    rectified-flow Euler×24, CFG 5)
                                                                          │
                                                                          ▼  latent (64×194)
                                                              Oobleck VAE decoder
                                                                          │
                                                                          ▼  stereo wav @ 44.1 kHz
```

The encoders run **once**; the DiT runs **24×** (×2 for batch-CFG); the VAE decoder runs **once**.

## 3. Project layout & build strategy

```
thinksound/
├─ CMakeLists.txt                # links against the prebuilt ggml in /devel/tools/llama.cpp
├─ docs/                         # this plan + architecture + conversion
├─ convert/
│  ├─ convert_thinksound.py      # thinksound_light.ckpt   -> GGUF (DiT)
│  ├─ convert_vae.py             # vae.ckpt                -> GGUF (oobleck decoder)
│  ├─ convert_metaclip_text.py   # facebook/metaclip-h14   -> GGUF (text tower)
│  └─ dump_reference.py          # PyTorch -> .npz/.gguf golden tensors for parity tests
├─ src/
│  ├─ ggml-common/               # tensor loading, backend sched, helpers, wav writer
│  ├─ tokenizer/                 # T5 SentencePiece + CLIP BPE
│  ├─ t5.{h,cpp}                 # T5 encoder graph (or reuse libllama t5encoder)
│  ├─ metaclip_text.{h,cpp}      # CLIP text tower graph
│  ├─ mmdit.{h,cpp}              # MM-DiT graph + sampler loop
│  ├─ vae_decoder.{h,cpp}        # oobleck decoder graph
│  └─ main.cpp                   # CLI glue
└─ tests/                        # per-component parity harness
```

**ggml dependency.** Link against the already-built ggml at `/devel/tools/llama.cpp` (CUDA 13.3 build present) instead of vendoring/rebuilding ~2 GB of CUDA objects. CMake: `find_library(ggml …)` + `find_library(ggml-cuda …)` pointing at that tree's `build/`, include `ggml/include`. We pin to that checkout's commit and document it. (Alternative — add as a git submodule — is cleaner for distribution but costs a full CUDA rebuild; defer until the runtime works.)

**Why not reuse `libllama`?** Only the T5 encoder is reusable there; the DiT/VAE/CLIP-text are bespoke graphs. We use raw `ggml` + `ggml-backend` (the `tools/tts` example is the template for a standalone ggml audio tool), and optionally `libllama` solely for T5.

## 4. Milestones

Each milestone is independently **numerically validated** against a PyTorch golden dump (see §5). "Parity" = max-abs-err < 1e-3 (f32 path) / acceptable f16 drift, and — for audio — a listenable match.

### M0 — Scaffolding & GGUF conversion (no inference)
- CMake project linking ggml; a `tensor-dump` utility that loads a GGUF and prints tensor names/shapes/dtypes.
- `convert_vae.py` and `convert_thinksound.py`: load the `.ckpt`, **fold weight-norm** (`weight_g`/`weight_v` → `weight`), strip EMA/optimizer wrappers, write GGUF with hparams metadata.
- **Done when:** GGUF files load in C++ and every expected tensor (per `ARCHITECTURE.md` tables) is present with correct shape. No math yet.

### M1 — Oobleck VAE decoder  ⭐ first audible milestone
- Implement the decoder graph: initial `conv1d(64→2048,k7)` → 5 × `DecoderBlock` (SnakeBeta → `conv_transpose_1d` → 3 dilated residual units) → final SnakeBeta → `conv1d(128→2,k7,no-bias)`.
- Primitives: SnakeBeta = `x + (1/exp(β))·sin²(exp(α)·x)`; dilated `ggml_conv_1d`; `ggml_conv_transpose_1d`; `group_norm(1,·)` inside residual units.
- **Validate:** dump a real 64×194 latent from PyTorch `predict.py`; decode in C++; compare waveform (and write the wav — it should sound like the reference). This proves conv/snake/transpose-conv mechanics before tackling the DiT.

### M2 — MM-DiT single forward step
- Implement: audio input proj (`conv1d k3` + SiLU + ConvMLP), timestep embedder (v2: freq-size 1024, max_period 1), conditioner preprocessing (clip/sync/text/t5 projections, sync pos-emb + nearest-exact interpolation, text concat 77+77→154), global cond, the **7 JointBlocks** (3-stream joint attention, RMS-QK-norm, interleaved RoPE on latent+clip), the **14 fused single-blocks**, FinalBlock.
- Feed PyTorch-dumped conditioning + a fixed latent + fixed `t`; compare the predicted flow tensor (64×194). This is the make-or-break correctness milestone (RoPE layout, modulate, joint-attn split).
- **Validate:** one-step flow parity, then the full 24-step Euler loop with CFG=5 vs PyTorch latent.

### M3 — End-to-end with dumped text features
- Wire M2 → M1: run the 24-step sampler then decode → wav, using **pre-extracted** `t5_features`/`metaclip_text_features` from PyTorch (`.npz`).
- **Validate:** full audio parity against `predict.py` output for a fixed seed. At this point the generative core is done; only the text front-end remains.

### M4 — MetaCLIP text encoder
- CLIP text tower (vocab 49408, width 1024, 24 layers, 16 heads, causal mask, ctx 77) producing both per-token (77×1024) and pooled→`text_projection`(1024) outputs. CLIP BPE tokenizer.
- **Validate:** token-ids + both feature tensors vs HF `facebook/metaclip-h14-fullcc2.5b`.

### M5 — T5 encoder
- Option A (reuse): convert `google/t5-v1_1-xl` via llama.cpp `convert_hf_to_gguf.py`; drive `libllama` T5ENCODER to get `last_hidden_state` (77×2048). Option B (native): implement T5 encoder as a ggml graph (relative-position bias, RMSNorm, gated-GELU FFN) for a dependency-free binary. Start with A; keep B as fallback.
- SentencePiece tokenizer (`spiece.model`), max_length 77, pad to max.
- **Validate:** hidden-state parity vs HF `T5EncoderModel`.

### M6 — Full text→audio integration + CLI
- Replace dumped features with live M4/M5 outputs; full CLI, seed control, duration handling (recompute seq lengths: `latent_seq_len = round(44100/64/32·dur)`, `clip_seq_len = 8·dur`, `sync_seq_len = 24·dur`).
- **Validate:** end-to-end vs the Python reference for several prompts/seeds.

### M7 — Quantization & perf
- Quantize DiT + T5 linears (Q8_0, Q6_K, Q4_K); keep norms/biases/conv-snake/embeddings at f16/f32. Measure quality drift (audio aesthetics + spectral distance) and speed/VRAM. VAE decoder likely stays f16 (small, quality-sensitive). See `GGUF_CONVERSION.md` §quantization.
- CUDA backend offload, KV-less so memory is modest; target both GPUs.

## 5. Validation harness (cross-cutting, build first)

`convert/dump_reference.py` patches the reference `predict.py` to dump, for a fixed seed/prompt, every boundary tensor to `.npz`/GGUF:
`t5_features`, `metaclip_text/global_features`, the conditioner outputs, `t_embed(t)`, the flow after step 0 and after 24 steps, the final latent, and the decoded waveform. Each C++ milestone diffs against these. A tiny `tests/parity.cpp` loads a golden GGUF + a C++ output and reports max/mean abs error. **This is the backbone of correctness — built in M0.**

## 6. GGML op coverage (all native; details in ARCHITECTURE.md §GGML notes)

| Need | ggml op | Risk |
|---|---|---|
| Linear, attention (SDPA), softmax, SiLU/GELU | matmul, soft_max, unary | none |
| RMSNorm (QK-norm), LayerNorm-no-affine | `rms_norm`, `norm` | none |
| Interleaved RoPE (FLUX 2×2) | `rope` normal mode + `freq_scale`, **or** precomputed cos/sin tables | **medium** — validate layout in M2 |
| SnakeBeta | `sin`,`sqr`,`exp`,`div`,`add` | low |
| Conv1d (dilated) / ConvTranspose1d | `conv_1d` (has dilation), `conv_transpose_1d` (CUDA kernel present) | low |
| GroupNorm(1,C) | `group_norm` | low |
| nearest-exact interpolate (sync→latent) | `upscale`/`interpolate` mode check | low |
| Batch-CFG | graph-level: duplicate batch, learned-empty uncond half, combine | low |

## 7. Risks & open questions (to confirm against weights/reference)

1. **RoPE layout parity** — `apply_rope` rotates *adjacent* pairs with a precomputed `[cos,-sin,sin,cos]`. Must confirm it equals `ggml_rope` "normal" mode (incl. the per-stream `freq_scaling` on the clip stream); else use explicit cos/sin tables. *Resolved in M2.*
2. **Checkpoint key nesting** — exact prefixes in `thinksound_light.ckpt` (e.g. `model.model.*`, `pretransform.*`, EMA copies). Confirmed only once we download & introspect; conversion script written defensively. *See GGUF_CONVERSION.md.*
3. **`thinksound_light` vs full provenance** — which weights are EMA/inference; verify the light ckpt is directly `load_state_dict`-able as `predict.py` implies.
4. **Conditioner wrapper mapping** — `get_conditioning_inputs` → `MMmodule.forward` arg order (esp. `metaclip_global_text_features` vs `metaclip_text_features`). *From ARCHITECTURE.md conditioners section; verify with a dump.*
5. **T5 reuse vs native** — whether `libllama` T5ENCODER reproduces `T5EncoderModel.last_hidden_state` exactly (it’s built for seq2seq). Fallback: native ggml T5.
6. **Duration generality** — reference hardcodes seq lengths from duration; verify non-9s durations and that RoPE/pos-emb recompute correctly.

## 8. Rough effort

| Phase | Relative effort |
|---|---|
| M0 conversion + harness | M |
| M1 VAE decoder | M |
| M2 MM-DiT | **L** (the bulk) |
| M3 integrate core | S |
| M4 MetaCLIP-text | M |
| M5 T5 | S (reuse) / M (native) |
| M6 CLI + integration | M |
| M7 quantization | M |

Recommended order is strictly M0→M7; M1 (audible VAE) and M2 (DiT parity) are the highest-information checkpoints and should gate everything after.
