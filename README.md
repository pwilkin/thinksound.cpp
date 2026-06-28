# thinksound.cpp

A standalone **C++ / GGML** runtime for [ThinkSound](https://huggingface.co/FunAudioLLM/ThinkSound)
(FunAudioLLM, NeurIPS 2025) — **text → sound-effect** generation with **no Python / PyTorch at
inference time**.

> **Status: complete & numerically verified.** A raw text prompt is turned into a 44.1 kHz stereo
> `.wav` entirely in C++/GGML, matching the PyTorch reference to ~1e-3 on the waveform and
> **1.0000 spectral correlation**. The full network stack (tokenizers → encoders → MM-DiT → VAE)
> runs on **CUDA**. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how it's built and
> [`docs/PROGRESS.md`](docs/PROGRESS.md) for the build log.

```sh
ts-generate \
  --caption "heavy rain on a metal roof" \
  --cot     "Heavy rain falls steadily on a metal roof, with distant thunder rumbling." \
  -o rain.wav
```

## What it does

Takes a text prompt — a short **caption** plus a longer **chain-of-thought description** — and
produces a 44.1 kHz stereo sound effect:

```
                tokenize            encode                 generate (24-step       decode
                (bit-exact)         (CUDA)                  rectified flow, CFG 5)  (CUDA)
  raw text ──► CLIP-BPE + T5  ──► MetaCLIP-text  ──►  MM-DiT  ─────────────────►  oobleck  ──► 44.1 kHz
               unigram            + T5-v1.1-xl         (7 joint + 14 fused)        VAE          stereo wav
```

Everything after the text prompt is pure GGML running on the GPU. There is no video / Synchformer /
VideoLLaMA2 path — absent-video conditioning uses the model's own learned empty embeddings, exactly
as the reference does for text-only generation.

## Highlights

- **Pure C++/GGML inference** — no torch, no Python, ~1.7k lines of runtime code.
- **Full GPU pipeline** — tokenizers (CPU, trivial), then MetaCLIP-text, T5-v1.1-xl, the MM-DiT,
  and the oobleck VAE decoder all run on CUDA.
- **Bit-exact tokenizers** — CLIP byte-level BPE and T5 SentencePiece-unigram, validated
  token-for-token against HuggingFace.
- **Numerically validated** against PyTorch at every stage (VAE 1e-6, DiT flow ~2e-5, end-to-end
  wav rel-L2 ~1e-3, spectral correlation 1.0000).
- **Quantization** — F32, **BF16** (near-lossless, recommended), and **Q8_0** GGUF variants of the
  DiT and T5, produced by an offline GGUF→GGUF quantizer (no checkpoint needed to re-quantize).

## Quick start

### 1. Build

The runtime links a **prebuilt** GGML from a llama.cpp checkout (CUDA 13.3 here):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`CMakeLists.txt` points at the GGML libraries under `/devel/tools/llama.cpp/build/bin`. Adjust
`GGML_DIR` / the library path there if your llama.cpp lives elsewhere.

### 2. Convert weights to GGUF (one-time)

From the official checkpoints (`thinksound_light.ckpt`, `vae.ckpt`) plus the T5 / MetaCLIP encoders:

```sh
python convert/convert_thinksound.py  --ckpt thinksound_light.ckpt --out dit-f32.gguf
python convert/convert_vae.py         --ckpt vae.ckpt              --out vae-f32.gguf
python convert/convert_t5.py          --model google/t5-v1_1-xl    --out t5-f32.gguf
python convert/convert_metaclip.py    --model facebook/metaclip-h14-fullcc2.5b --out metaclip-text-f32.gguf
python convert/convert_tokenizers.py  --out-clip clip-tokenizer.gguf --out-t5 t5-tokenizer.gguf

# optional: quantize an existing GGUF without touching the checkpoint
python convert/quantize_gguf.py --type bf16 --in dit-f32.gguf --out dit-bf16.gguf
python convert/quantize_gguf.py --type q8_0 --in t5-f32.gguf  --out t5-q8.gguf
```

See [`docs/GGUF_CONVERSION.md`](docs/GGUF_CONVERSION.md) for the full key-mapping and weight-norm
folding details.

### 3. Generate

```sh
ts-generate \
  --caption "a dog barking" \
  --cot     "A medium-sized dog barks several times in a quiet room." \
  --duration 9 \
  --dit  dit-bf16.gguf  --t5 t5-bf16.gguf  --clip metaclip-text-f32.gguf \
  --vae  vae-f32.gguf   --clip-tok clip-tokenizer.gguf --t5-tok t5-tokenizer.gguf \
  -o dog.wav
```

`--duration <sec>` (default 9, 1–30) sets the clip length — the latent/clip/sync sequence
lengths are derived from it (`T=round(44100/2048·D)`, `clip_S=8·D`, `sync_S=24·D`). `--steps`,
`--cfg`, and `--seed` tune the sampler. `ts-generate --ref golden_e2e.gguf …` reuses the golden's
noise (pinning the 9 s length) and prints latent/wav parity against the PyTorch dump.

## Models & quantization

| Network        | Source                                   | F32   | BF16  | Q8_0  |
|----------------|------------------------------------------|-------|-------|-------|
| MM-DiT         | `thinksound_light.ckpt` (DiT submodule)  | 4.8 G | 2.4 G | 3.7 G |
| T5-v1.1-xl     | `google/t5-v1_1-xl` (encoder)            | 4.6 G | 2.5 G | 1.4 G |
| MetaCLIP-text  | `facebook/metaclip-h14-fullcc2.5b`       | F32   | —     | —     |
| oobleck VAE    | `vae.ckpt` (decoder)                     | F32   | —     | —     |

**BF16 is the recommended default** — spectral correlation 0.9999 vs F32, roughly half the memory.
Q8_0 is also viable (spectral correlation 0.9987) and is the smallest T5. Quality ranking on the
waveform: **F32 ≳ BF16 ≫ Q8_0**, all perceptually indistinguishable. See
[`docs/PROGRESS.md`](docs/PROGRESS.md) for the full assessment.

## Verification

Every stage is checked against the PyTorch reference with golden-tensor dumps (`convert/dump_*.py`):

| Stage                       | Metric            | Result            |
|-----------------------------|-------------------|-------------------|
| Tokenizers (CLIP + T5)      | token ids         | bit-exact         |
| Oobleck VAE decode          | rel-L2            | ~1e-6             |
| MM-DiT single flow step     | rel-L2            | ~2e-5             |
| **End-to-end (text → wav)** | wav rel-L2        | ~1e-3             |
| **End-to-end (text → wav)** | spectral corr     | **1.0000**        |

The all-GPU waveform rel-L2 (~1.4e-2) is slightly higher than all-CPU (~1.4e-3) because cuBLAS 13
uses TF32 for f32 GEMMs — phase-level only; the spectrum is identical (corr 1.0000).

## Repository layout

```
src/
  common/      ts_model (GGUF loader + backend streaming), ts_backend (device pick), ts_wav
  tokenizer.*  CLIP byte-level BPE + T5 unigram (Viterbi)
  clip_text.*  MetaCLIP text tower         t5_encoder.*  T5-v1.1-xl encoder
  mmdit.*      MM-DiT (the core)           vae_decoder.* oobleck VAE decoder
  tools/       ts-generate + per-stage test/parity/dump tools
convert/       checkpoint→GGUF converters, GGUF→GGUF quantizer, golden dumpers
docs/          ARCHITECTURE (model spec), IMPLEMENTATION_PLAN, GGUF_CONVERSION, PROGRESS
ARCHITECTURE.md  as-built runtime architecture (start here for the code)
```

## Documents

| Doc | Contents |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | **As-built runtime**: code map, data flow, backend strategy, layout conventions, key implementation decisions. Start here to understand the code. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Implementation-ready **model spec** of every component (shapes, tensors, math, ggml-op mapping), reverse-engineered from the PyTorch reference. |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Phased plan M0–M7, project layout, validation harness, risks. |
| [`docs/GGUF_CONVERSION.md`](docs/GGUF_CONVERSION.md) | Checkpoint→GGUF mapping, weight-norm folding, GGUF metadata, quantization analysis. |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Running build log / resume state with all parity numbers. |

## Scope

- **Text → SFX only** (no video). Decided 2026-06-16.
- Target checkpoint: `thinksound_light.ckpt` (5.73 GB) + `vae.ckpt` (2.52 GB), from HF
  `liuhuadai/ThinkSound`.
- Encoders: `google/t5-v1_1-xl`, `facebook/metaclip-h14-fullcc2.5b` (text tower).
- Builds against the GGML in `/devel/tools/llama.cpp` (CUDA 13.3).

## License

The reference model and weights are under their respective upstream licenses (see the ThinkSound
repo). This runtime is an independent reimplementation.
