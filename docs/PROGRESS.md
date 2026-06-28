# Build progress / resume state

_Updated 2026-06-16. Autonomous /loop building toward a verified end-to-end text→SFX._

## FINAL STATE — whole pipeline on GPU, verified (supersedes stale notes below)
- **All five networks run on CUDA**: tokenizers (CPU, trivial) → MetaCLIP-text, T5, MM-DiT, oobleck VAE all on GPU. End-to-end ~29 s, **spectral-corr 1.0000**, wav rel-L2 1.4e-2 (DiT cuBLAS-13 TF32; perceptually identical).
- **VAE on GPU**: transpose-conv via the **zero-stuff + conv1d** identity (`col2im_1d` has no CUDA kernel; `ggml_conv_transpose_1d`'s kernel is O(out·IC·OC·T_in) → hangs on ×2048). Kernel flipped + channel-swapped at conversion (`convert_vae.py:to_conv1d_kernel`, ggml ne=[K,IC,OC]). Zero-stuff avoids the `ggml_pad` `ne1→gridDim.y≤65535` trap by padding the stride axis + permute/cont. VAE GPU parity rel-L2 6.8e-4 vs golden. See `ARCHITECTURE.md §5`.
- **CUDA soft_max root-fixed environmentally**: removed the Debian `nvidia-cuda-dev`/`libcudart12` package that shadowed the 13.3 cudart (header/runtime ABI mismatch → garbage `smpbo`). Cleared all stale `CUDA_*_LIBRARY` CMake cache entries → re-resolve to /usr/local/cuda-13.3 → `ldd libggml-cuda` shows libcudart.so.13, `test-backend-ops SOFT_MAX` 212/212. **My earlier ggml-cuda.cu patch was reverted — llama.cpp tree is clean.** (So T5 + CLIP on GPU need no patch.)
- **Memory**: `ts-generate` loads each network in its own scope and frees before the next (all 4 weights + VAE intermediates OOM a 16 GB card otherwise). Peak VRAM = one stage, not the sum.
- Docs: top-level `README.md` (overview/usage) + `ARCHITECTURE.md` (as-built runtime). `docs/ARCHITECTURE.md` remains the model spec.

## Done & verified
- **M0** — env + weights + converters + build + harness.
  - venv: `/media/ilintar/D_SSD/thinksound/.venv` (py3.12, torch 2.12 cpu). Run reference scripts with `PYTHONPATH=/tmp/ThinkSound` and **stub `k_diffusion`** (`sys.modules['k_diffusion']=types.ModuleType(...)`) to dodge broken `clip`/`pkg_resources` transitive deps.
  - weights: `/media/ilintar/D_SSD/thinksound/ckpts/{thinksound_light.ckpt,vae.ckpt}`.
  - GGUF: `gguf/vae-f32.gguf` (78M decoder), `gguf/dit-f32.gguf` (1.277B). Converters in `convert/`.
  - C++ builds against prebuilt ggml at `/devel/tools/llama.cpp/build` (commit 911b67a60). `cmake -S . -B build && cmake --build build -j4`.
  - tools: `ts-gguf-dump`, `ts-vae_decode`.
- **M1** — Oobleck VAE decoder in ggml. **PARITY rel_L2 = 1e-6** vs PyTorch (`ts-vae_decode` on `golden/golden_vae.gguf`). Audible. `src/vae_decoder.{h,cpp}`.

## In progress
- **M2** — MM-DiT + rectified-flow sampler.
  - `gguf/dit-f32.gguf` ready.
  - `convert/dump_dit_golden.py` running in background (`/media/ilintar/D_SSD/thinksound/golden_dit.log`, ~10-15 min on CPU) → `golden/golden_dit.gguf` with {noise, clip_f, sync_f, text_f, t5_features, metaclip_global_text_features, flow0, latent, wav_ref}. Uses text-only path (video=learned empties, text/t5=seeded random).
  - NEXT: write `src/mmdit.{h,cpp}` (input projs, v2 timestep embed [freq 1024, max_period 1], FLUX interleaved RoPE on latent+clip, 7 JointBlocks [3-stream joint attn + adaLN + conv-SwiGLU-FFN], 14 fused blocks, FinalBlock, batch-CFG). Verify `flow0` then full `latent` then decode→`wav_ref`.

## Key technical learnings (reuse)
- **`/devel/tools/acestep.cpp`** is a ggml DiT-audio model with the **same oobleck VAE** — `src/vae.h` and `src/dit.h` are direct references for snake / conv / transpose-conv / DiT idioms.
- `ggml_conv_1d` forces F16 im2col (asserts F16 kernel). For f32 parity, build conv from explicit `ggml_im2col(...,GGML_TYPE_F32)` + `mul_mat` (see `src/vae_decoder.cpp::conv1d`).
- `ggml_conv_transpose_1d` asserts `p0==0` (no padding) → unusable for oobleck. Use **GEMM + `ggml_col2im_1d`**: pack transpose weight to `[IC, K*OC]` (i=oc*K+k) in the converter (`convert_vae.py::pack_convtranspose`), then `mul_mat(w, xᵀ)`→`col2im_1d(stride, OC, pad)`.
- ggml audio layout: `[T, C]` (ne[0]=time, ne[1]=channels). conv kernel ne `[K,IC,OC]`; data `[T,IC]`.
- SnakeBeta: `x + sin²(exp(α)·x)·exp(−β)`, α/β per-channel `[C]`→reshape `[1,C]`.
- DiT ckpt prefix `model.model.*`; VAE `pretransform.model.*` (also in light ckpt) / `vae.ckpt` `autoencoder.*`. No EMA in light ckpt. mm_unchang conditioners are weightless.

## Done since (all verified vs PyTorch)
- **M2 sampler + M3**: CFG + 24-step rectified-flow Euler + VAE decode. flow0 rel_L2 1.1e-3, latent 6.7e-4, wav 1.8e-3. (`ts-dit_sample`)
- **M4/M5 encoders**: T5-v1.1-xl (`src/t5_encoder.cpp`, `gguf/t5-f32.gguf`) rel_L2 2.2e-3; MetaCLIP-text (`src/clip_text.cpp`, `gguf/metaclip-text-f32.gguf`) per-token 1.6e-4, global 1.5e-4. (`ts-encoders_test`). **KEY FIX:** both HF encoders apply the tokenizer padding attention_mask (CLIP also causal); must replicate or only pooled@eos matches. Goldens: `golden_text.gguf` (caption "a dog barking", cot "...").

## CRITICAL BUG FIXED (LayerNorm eps)
The DiT's affine-free `nn.LayerNorm` uses PyTorch default **eps=1e-5**, NOT 1e-6. Symptom: full text→audio diverged (latent 14%, wav 21%) while every component passed individually and the *synthetic-feature* DiT golden matched at 3e-4. Root cause: real text features contain **padding tokens with ~0 variance**; for those, layernorm `(x-mean)/sqrt(var+eps)` is eps-dominated, so 1e-6 vs 1e-5 (√10≈3.16×) corrupts every padding token → wrong text q/k/v → wrong joint attention. Synthetic random features have no padding → no near-zero-variance tokens → masked the bug. Fix in `src/mmdit.cpp::layernorm` (1e-5). After fix: e2e staged flow rel_L2 **1e-5**. Lesson: match norm eps exactly; test with REAL (padded) inputs, not just random. RMSNorm q/k kept at 1e-6 (not padding-sensitive).

## CUDA soft_max bug FIXED — T5 + CLIP + DiT now all on GPU
Root cause (found via compute-sanitizer + instrumenting `softmax.cu`): `ggml_cuda_info().smpbo` was garbage (`4294967297 = 0x100000001`) because it's read from the `cudaDeviceProp.sharedMemPerBlockOptin` struct field, and **ggml-cuda was compiled with CUDA 13.3 headers but links libcudart.so.12** → struct layout mismatch → that field read from the wrong offset (VRAM/cc sit at stable offsets, so only soft_max broke). Fix in `ggml/src/ggml-cuda/ggml-cuda.cu` (~line 290): read the limit via `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` (stable single-value ABI) instead of the struct field. `test-backend-ops -o SOFT_MAX -b CUDA1` now **212/212 pass**. Rebuild: `cmake --build /devel/tools/llama.cpp/build --target ggml-cuda`. (Real ggml fix — benefits all the user's CUDA usage; the deeper cause is a mismatched CUDA toolkit/runtime in their build.)

Result: **T5, CLIP, DiT all run on GPU** (T5/CLIP reverted to `ts_backend_init()`); full pipeline wav rel_L2 **1.29e-3** (better than the old hybrid 1.86e-3 — T5 uses exact soft_max now, not CPU). **Only the VAE stays on CPU** (`ggml_col2im_1d` genuinely has no CUDA kernel — a separate gap, fixable via `conv_transpose_1d`(p0=0)+crop; VAE is one decode, negligible).

## GPU (M7, earlier) — historical: DiT on CUDA, encoders+VAE on CPU
`src/common/ts_backend.h` picks the max-memory GPU (CUDA1 = RTX 5060 Ti 16GB); `ts_model::load_backend` streams weights into a backend buffer. Per-model: **DiT → GPU**, encoders+VAE → CPU (`ts_backend_init(false)`; `TS_FORCE_CPU=1` forces all-CPU; `TS_BACKEND=CUDA0/CPU` overrides).

**This ggml build's CUDA backend gaps (commit 911b67a60), discovered the hard way:**
- **Standalone `ggml_soft_max` is BROKEN on CUDA** (even [128,10] → "invalid argument"; llama.cpp's GPU path uses fused flash-attn so standalone soft_max is untested). → DiT attention rewritten to **`ggml_flash_attn_ext`** (k/v cast F16, `set_prec(F32)`), works great: flow rel_L2 2e-5, full pipeline wav rel_L2 1.86e-3.
- CUDA flash-attn mask requires `mask->ne[2]==1` (broadcast over heads) + 16-byte-aligned strides → **T5's per-head relative bias can't use it** → T5 stays CPU.
- **`ggml_col2im_1d` has no CUDA kernel** → VAE transpose-conv stays CPU (or re-do as `conv_transpose_1d`(p0=0)+crop with an unpacked `[K,OC,IC]` weight).
- GGML device enum here: CUDA cards report type=2 (IGPU), not GPU(1) — select GPU||IGPU.
- CUDA binary ops need contiguous operands → `chunk()` must `ggml_cont` the adaLN views.

Path to full-GPU (future): CLIP→flash-attn with padded broadcast mask; VAE→conv_transpose_1d+crop; T5 likely stays CPU (per-head bias).

## DONE — full standalone text→SFX (M6 complete)
- **Tokenizers** (`src/tokenizer.cpp`, `gguf/{clip,t5}-tokenizer.gguf` via `convert/convert_tokenizers.py`): CLIP byte-level BPE + T5 SentencePiece unigram (Viterbi). **Bit-exact** vs HF on the test caption/cot (`ts-tok_test`).
- **`ts-generate`** is now the standalone CLI: `--caption/--cot` raw text → tokenize → encode → DiT(GPU) → VAE → wav. Verified vs `golden_e2e.gguf`: latent rel_L2 8.3e-4, wav 1.86e-3. Generates novel prompts (e.g. "heavy rain on a metal roof").

## Q8_0 quantization — VIABLE (assessed)
`convert/quantize_gguf.py f32.gguf q8.gguf` (or `--quant q8_0` in the converters). Quantizes 2D linear weights to Q8_0; keeps convs (3D, fed to im2col), norms/biases (1D), embeddings & empties f32. **No C++ change** — ggml `mul_mat` auto-dispatches on weight type (works on CUDA + CPU).
- Size: DiT 4.8→3.7GB (23%; convs stay f32), T5 4.6→1.4GB (70%). Total weights ~11→6GB.
- Speed: 42.8s→31.3s (27% faster); peak RAM 8.9→5.6GB (37% less).
- Quality (vs PyTorch ref): waveform rel_L2 6.8% (phase-level, misleading) BUT **spectral-corr 0.9987, envelope-corr 0.9995, log-spec-dist 0.318 vs 0.235 f32** → perceptually identical. Attribution: T5-Q8 4.6% wav (bigger), DiT-Q8 2.6%.
- To compress DiT convs too (~60% DiT saving): store conv weight pre-reshaped `[K*IC, OC]` and pass a shape-only kernel to im2col, OR a conv-friendly quant. Q4_K would be the next step to test.

## BF16 (recommended near-lossless) + why T5 is CPU
**T5-on-GPU is blocked by a real ggml bug**, not my code: `test-backend-ops -o SOFT_MAX -b CUDA1` (llama.cpp's own test) FAILS ("invalid argument") on commit 911b67a60 — standalone CUDA soft_max is broken. T5's per-head relative bias also can't use CUDA flash-attn (it requires `mask->ne[2]==1`). So T5 (and VAE col2im) stay CPU. The DiT only works on GPU because it uses mask-less flash-attn.

**Conv weights are now quantizable**: `conv1d_f32` was restructured to `mul_mat(weight, cols)` (weight = first operand → can be bf16/quantized; im2col only needs its shape). Numerically identical for f32 (removes a transpose). `quantize_gguf.py --type bf16` now covers convs too.

**f32 vs bf16 vs Q8** (dog-barking, vs PyTorch reference):
| | DiT | T5 | spec-corr | env-corr | lsd | time |
|---|---|---|---|---|---|---|
| f32 | 4.8GB | 4.6GB | 1.0000 | 1.0000 | 0.235 | 39.2s |
| **bf16 (full)** | **2.4GB** | **2.5GB** | **0.9999** | **1.0000** | 0.266 | 37.1s |
| Q8 | 3.7GB | 1.4GB | 0.9987 | 0.9995 | 0.318 | 37.1s |

BF16 = perceptually identical (spec-corr 0.9999), true 50% DiT (convs incl). Q8 = smaller T5 but DiT partial (convs f32) + slightly more loss. Both fine. For max DiT-Q8 you'd pre-reshape conv weights to `[K·IC, OC]` (ne0 block-aligned) + feed im2col a shape-only tensor.

## Optional future polish (not blocking)
- Full-GPU encoders/VAE (CLIP→flash-attn padded mask; VAE→conv_transpose_1d+crop; T5 likely stays CPU).
- Quantization (Q8/Q4 DiT+T5) for memory/speed; f16 weights to fit the 10GB card.
- Embed tokenizers + hparams into the model GGUFs for single-file distribution.
- CLIP regex pre-tokenization is simplified (whitespace+punct); exact for typical captions, may differ on unusual punctuation.
