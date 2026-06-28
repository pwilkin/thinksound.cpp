# thinksound.cpp — Runtime Architecture (as built)

This document describes **how the C++/GGML runtime is organized and why**. For the exhaustive
per-component *model* spec (every tensor, shape and equation reverse-engineered from the PyTorch
reference) see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). This file is the map you read first
to understand the code.

---

## 1. Pipeline

A single `ts-generate` invocation runs five networks in sequence. Everything after tokenization is
GGML compute graphs executed on the GPU.

```
 raw text (caption + chain-of-thought)
      │
      ▼  tokenizer.*            CPU   — CLIP byte-level BPE + T5 unigram, bit-exact vs HF
 token ids
      │
      ├──────────────► clip_text.*   CUDA  — MetaCLIP text tower → pooled 1024-d global + tokens
      └──────────────► t5_encoder.*  CUDA  — T5-v1.1-xl encoder   → 2048-d sequence features
      │
      ▼  mmdit.*                CUDA  — MM-DiT, rectified-flow 24-step Euler, batch-CFG (scale 5)
 audio latent  [64, T]
      │
      ▼  vae_decoder.*          CUDA  — oobleck decoder, ×2048 upsample
 waveform  [2, T·2048]
      │
      ▼  ts_wav.*               CPU   — interleave + write 44.1 kHz stereo PCM16
 .wav
```

Conditioning for the absent video stream uses the DiT's own learned `empty_clip_feat` /
`empty_sync_feat` parameters, so the text-only path is exactly what the reference runs when no video
is supplied — not an approximation.

---

## 2. Code map

| File | Role | Key entry points |
|---|---|---|
| `common/ts_model.{h,cpp}` | GGUF loader + weight streaming | `load()` (host), `load_backend(path, backend)` (stream into device buffer), `get_data()`, `get_arr_f32()`, `get_arr_str()` |
| `common/ts_backend.h` | Backend/device selection | `ts_backend_init(bool want_gpu=true)` |
| `common/ts_wav.{h,cpp}` | PCM16 WAV writer | `ts_wav_write()` |
| `tokenizer.{h,cpp}` | Text → token ids | CLIP BPE encoder, T5 unigram (Viterbi) encoder |
| `clip_text.{h,cpp}` | MetaCLIP text tower | `ts_clip_text::load/encode` |
| `t5_encoder.{h,cpp}` | T5-v1.1-xl encoder | `ts_t5::load/encode` |
| `mmdit.{h,cpp}` | MM-DiT + sampler | `ts_mmdit::load`, `predict_flow`, `sample` |
| `vae_decoder.{h,cpp}` | Oobleck VAE decoder | `ts_vae::load/decode` |
| `tools/generate.cpp` | `ts-generate` CLI | end-to-end driver + `--ref` parity |
| `tools/*_test.cpp`, `dit_sample.cpp`, `vae_decode.cpp` | per-stage harnesses | numeric checks vs golden |
| `convert/*.py` | checkpoint→GGUF, GGUF→GGUF quant, golden dumps | run with `PYTHONPATH=…/ThinkSound` |

Each network is a small self-contained class: `load()` reads a GGUF into a backend buffer and keeps
a `name → ggml_tensor*` map; `encode()`/`decode()`/`sample()` builds a fresh compute graph per call,
allocates it with `ggml_gallocr`, sets the inputs, and runs `ggml_backend_graph_compute`.

---

## 3. Backend strategy

`ts_backend_init()` (in `common/ts_backend.h`) picks the compute device once per network:

- Enumerates ggml backend devices and selects the **GPU with the most free memory**. CUDA cards
  enumerate with device type `GGML_BACKEND_DEVICE_TYPE_IGPU` (=2) in this ggml build, so both `GPU`
  and `IGPU` types are accepted.
- Falls back to CPU if no GPU is present or `want_gpu == false`.
- Env overrides: `TS_FORCE_CPU=1` forces CPU everywhere; `TS_BACKEND=CUDA0|CUDA1|CPU` pins a
  specific device.

All five networks request the GPU. Weights are streamed straight into the device buffer at load time
(`ggml_backend_alloc_ctx_tensors` + per-tensor `fread` + `ggml_backend_tensor_set`), so the host
never holds a second full copy.

The only operation without a native CUDA kernel that the pipeline historically needed was
`col2im_1d` (used by the original VAE transpose-conv). That is avoided entirely — see
[§5 transpose conv](#5-transposed-convolution-the-vae-gpu-path) — so the whole pipeline is GPU-resident.

**Sequential staging.** The four networks together (~10 GB of weights) plus the VAE's multi-GB
im2col intermediates would OOM a 16 GB card. Since the stages run strictly in order
(encode → sample → decode), `ts-generate` loads each network in its own scope and frees it before
the next, so peak VRAM is *one* stage's weights + intermediates, not the sum. Outputs cross scope
boundaries as small host vectors (encoder features, the audio latent).

---

## 4. Layout & convolution conventions

GGML tensors are column-major in `ne[]` (fastest dim first). Two conventions matter throughout:

- **Audio/sequence tensors are `[time, channels]`** (`ne[0]=time`, `ne[1]=channels`). This matches
  how `ggml_conv_*` and `mul_mat` want their operands and avoids transposes between stages.
- **Linear/attention features are `[dim, tokens]`** (`ne[0]=feature`).

### 1-D convolution

`ggml_conv_1d` forces an **F16** im2col kernel, which both blocks quantized conv weights and loses
precision. The runtime builds convolution explicitly instead, with the **weight as the first
`mul_mat` operand** so it can be F32 / BF16 / quantized while im2col only contributes shape:

```cpp
// mmdit.cpp / vae_decoder.cpp
ggml_tensor * conv1d_f32(ctx, w, x, stride, pad, dilation) {
    im   = ggml_im2col(ctx, w, x, stride, 0, pad, 0, dilation, 0, false, GGML_TYPE_F32); // [IC*K, OL, 1]
    cols = ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1]*im->ne[2]);                      // [IC*K, OL]
    wr   = ggml_reshape_2d(ctx, w, w->ne[0]*w->ne[1], w->ne[2]);                          // [IC*K, OC]
    return ggml_mul_mat(ctx, wr, cols);                                                  // [OC, OL]
}
```

`im2col` with `GGML_TYPE_F32` is fully supported on CUDA, so these convs run on the GPU at full
precision.

---

## 5. Transposed convolution (the VAE GPU path)

The oobleck decoder upsamples ×2048 through five transpose-conv stages (strides 8·8·4·4·2). Each is a
PyTorch `ConvTranspose1d` with kernel `K = 2·stride` and padding `p = stride/2`. Getting this onto
CUDA took ruling out two dead ends:

- **`col2im_1d`** (GEMM + overlap-add) is efficient but has **no CUDA kernel** — the original reason
  the VAE was stuck on CPU.
- **`ggml_conv_transpose_1d`** *has* a CUDA kernel, but it's O(out_len·IC·OC·T_in) — it rescans the
  whole input per output element. For the VAE's ×2048 upsample that's ~1e17 ops; it effectively
  hangs.

So the runtime uses the standard **zero-stuff + regular conv** identity: a stride-`s` transpose conv
equals inserting `s−1` zeros between input samples, then a stride-1 conv with the kernel flipped
along `K` and its input/output channels swapped. The regular conv is `pad + im2col + GEMM` — all fast
CUDA kernels. The flipped/swapped kernel is precomputed at conversion (`convert_vae.py`,
`to_conv1d_kernel`) and stored as ggml `ne = [K, IC, OC]`.

The one subtlety is the zero-stuffing. `ggml_pad` maps `ne1 → gridDim.y` (CUDA max 65535), and the
VAE's time dimension reaches ~200k — so padding time directly overflows the launch grid. Instead the
runtime pads the *stride* axis (small) with time kept in `ne0`/`gridDim.x`, then `permute`+`cont`
into the interleaved layout (the copy kernel uses a flat grid, so it's safe for long sequences):

```cpp
// vae_decoder.cpp — conv_t1d(w[K,IC,OC], x[T_in,IC], stride)
x3 = reshape(x, T_in, 1, IC);            //                              [T_in, 1, IC]
P  = ggml_pad(x3, 0, stride-1, 0, 0);    // pad stride axis (grid-safe)  [T_in, stride, IC]
Pp = cont(permute(P, 1,0,2,3));          // interleave time              [stride, T_in, IC]
uf = reshape(Pp, stride*T_in, IC);       // zero-stuffed                 [stride*T_in, IC]
u  = cont(view(uf, (T_in-1)*stride+1));  // drop trailing zeros
y  = conv1d(w, b, u, /*stride*/1, /*pad*/K-1-stride/2, /*dil*/1);     // [T_in*stride, OC]
```

`(T_in−1)·stride + 1` input length through a stride-1 conv with pad `K−1−p` yields exactly
`T_in·stride`. This runs the entire decoder on CUDA (VAE GPU rel-L2 6.8e-4 vs the golden; the gap
from CPU's 1e-6 is cuBLAS-13 TF32 on the GEMMs, perceptually irrelevant).

### SnakeBeta + weight-norm

Oobleck uses the SnakeBeta activation `x + sin²(exp(α)·x) · exp(−β)`, built from
`ggml_sin`/`ggml_sqr`/`ggml_exp`. Convolution weights are weight-normalized in the checkpoint; the
converter **folds** `g · v/‖v‖` into a single dense weight offline so inference is a plain conv.

---

## 6. MM-DiT details

The core (`mmdit.cpp`) is an MMAudio-derived multimodal DiT: depth 21 = **7 joint** (3-stream:
latent / clip / text) blocks + **14 fused** single-stream blocks, hidden 1024, 16 heads, head_dim 64.

- **Attention** uses `ggml_flash_attn_ext` (K/V cast to F16, `ggml_flash_attn_ext_set_prec(o,
  GGML_PREC_F32)`), which on CUDA is faster and more memory-frugal than an explicit
  `softmax(QKᵀ)·V`. The T5 and CLIP encoders instead use plain `ggml_soft_max(_ext)` because their
  attention carries per-head bias/masks that the flash-attn CUDA kernel doesn't accept
  (`mask->ne[2] != 1` → no kernel).
- **RoPE** is FLUX-style interleaved (ggml mode `GGML_ROPE_TYPE_NORMAL`), applied to Q/K.
- **Modulation** is adaLN: a timestep embedding drives per-block scale/shift/gate. The conditioning
  preprocessing (text/clip projections, empty-video embeddings) is computed once and cached across
  all 24 sampler steps.
- **FFN** is SwiGLU with a conv branch (`conv1d_f32`) and a linear branch.
- **CFG** is *batch* classifier-free guidance: conditioned and unconditioned passes are stacked into
  one batch of 2 and combined as `uncond + scale·(cond − uncond)` with `scale = 5`.
- **Sampler**: rectified-flow, 24-step uniform Euler from t=1→0 on the velocity field.

`ggml_chunk`/view outputs are wrapped in `ggml_cont` before feeding CUDA ops, which require
contiguous operands.

---

## 7. Encoders & tokenizers

- **Tokenizers** (`tokenizer.cpp`): CLIP byte-level BPE and T5 SentencePiece-unigram (Viterbi best
  path). Both validated token-for-token against HuggingFace; the merge ranks / unigram vocab+scores
  ship in `clip-tokenizer.gguf` / `t5-tokenizer.gguf`.
- **MetaCLIP text** (`clip_text.cpp`): pre-LN causal transformer, `gelu_quick`, `soft_max_ext` with a
  combined causal+padding mask, LayerNorm eps **1e-5**.
- **T5-v1.1-xl** (`t5_encoder.cpp`): relative-position attention bias (per-head), gated-GeLU FFN,
  RMSNorm, plain `soft_max`.

---

## 8. Quantization

`convert/quantize_gguf.py` re-quantizes an existing GGUF **without the checkpoint** (GGUF→GGUF),
emitting `--type {bf16, q8_0}`:

- **BF16** — per-element truncation, near-lossless (spectral corr 0.9999). Recommended default.
- **Q8_0** — block-of-32 8-bit. Smallest T5 (1.4 G); spectral corr 0.9987.

Because `conv1d_f32` puts the weight first in the `mul_mat`, conv weights are quantizable alongside
the linear weights. Norm/embedding/bias tensors are kept in F32.

---

## 9. Verification methodology

Numerical parity is enforced stage-by-stage against PyTorch golden dumps
(`convert/dump_{vae,dit,text,e2e}_golden.py`, run with `k_diffusion` stubbed and
`PYTHONPATH` pointed at the reference checkout):

1. **Tokenizers** → token ids must be bit-exact.
2. **VAE** (`ts-vae_decode --ref`) → rel-L2 ~1e-6.
3. **DiT** single step (`ts-dit_test`) → rel-L2 ~2e-5.
4. **End-to-end** (`ts-generate --ref`) → latent + wav rel-L2, plus a spectral-correlation check
   (the perceptually meaningful metric; corr 1.0000).

Two bugs found this way are worth remembering:

- **Affine-free LayerNorm eps must be 1e-5, not 1e-6.** Padding tokens have ~0 variance, so eps
  dominates their normalization; the wrong eps caused a ~14 % full-pipeline divergence that only
  appeared with *real padded* inputs (synthetic dense inputs hid it). Lesson: test with realistic
  padded batches.
- **CUDA `soft_max` "invalid argument".** ggml-cuda read `cudaDeviceProp.sharedMemPerBlockOptin` as
  garbage (`0x100000001`) because the CUDA backend was compiled with 13.3 headers but linked an old
  `libcudart.so.12` (struct-ABI mismatch). The real fix is environmental — remove the stray
  CUDA-12 runtime so the toolkit's own 13.3 cudart is linked; `test-backend-ops -o SOFT_MAX`
  then passes 212/212.

---

## 10. Build & dependencies

The runtime links a **prebuilt** GGML (no vendored copy). `CMakeLists.txt` builds a `ts_common`
static library (the loaders + all five networks) and links each `tools/*.cpp` into a `ts-*` binary.
CUDA support comes from the linked `libggml-cuda`; the only hard requirement is that GGML's CUDA
backend and the system `libcudart` are the **same major version** (see §9).
