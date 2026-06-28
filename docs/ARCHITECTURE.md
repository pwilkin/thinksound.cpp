# ThinkSound — Architecture Spec for the GGML Port (text → SFX path)

> Reverse-engineered from the PyTorch reference (`FunAudioLLM/ThinkSound`, cloned at `/tmp/ThinkSound`) by reading the source directly. Every shape/tensor/formula below is traced to code with `file:line`. Scope: the **text-to-sound-effects** path only (no video → Synchformer / VideoLLaMA2 / MetaCLIP-visual are omitted; absent-video conditioning uses the model's learned empty embeddings).
>
> All numeric seq-lengths are for the default **9 s** duration; they scale with duration (§9).

## 1. Overview & data flow

ThinkSound generates audio by **rectified-flow latent diffusion**. A multimodal DiT (the "MMmodule", adapted from MMAudio) predicts a velocity field over a 64-channel audio latent; a 24-step Euler integrator turns Gaussian noise into a latent; a Stable-Audio "oobleck" VAE decoder turns that latent into a stereo 44.1 kHz waveform. Text conditioning comes from two frozen encoders run once up front.

```
 caption ──► MetaCLIP text tower ──► global  [1,1024] ──────────────► text_cond_proj ─┐
       │                          └► per-token [1,77,1024] ─┐                          │
       │                                                     ├─ pad→2048, concat ──► text stream [1,154,1024]
 cot  ──► T5-v1.1-xl encoder ─────► [1,77,2048] ────────────┘                          │
                                                                                       ▼
 (no video) empty_clip_feat[1,1024]→[1,72,1024]  empty_sync_feat[1,768]→[1,216,768]   MM-DiT
                         │                              │                          (7 joint + 14 fused,
                         ▼                              ▼                           rectified flow)
                    clip stream                    sync stream                          │
                                                                       24× Euler step  │  v = f(x,t,cond)
 x₀ = randn[1,64,194] ───────────────────────────────────────────────────────────────►│  x += (t'-t)·v_cfg
                                                                                        ▼
                                                                            latent x [1,64,194]
                                                                                        │
                                                                         Oobleck VAE decoder (÷2048 upsample)
                                                                                        ▼
                                                                        stereo waveform [1,2,397312] ≈ 9.0 s
```

Encoders run **once**. The DiT runs **24×** with a **doubled batch** for classifier-free guidance (CFG). The VAE decoder runs **once**.

## 2. Components

| # | Component | Source weights | ~params | Reuse? | GGML-op risk |
|---|---|---|---|---|---|
| 1 | MetaCLIP text tower | `facebook/metaclip-h14-fullcc2.5b` (text only) | ~0.36 B | build fresh (small) | low (causal attn + final LN + proj) |
| 2 | T5-v1.1-xl encoder | `google/t5-v1_1-xl` | ~1.2 B | **reuse** llama.cpp `T5ENCODER` (or native) | low |
| 3 | MM-DiT ("MMmodule") | `thinksound_light.ckpt` | ~1.4 B | build fresh | **medium** (FLUX RoPE, joint attn, adaLN, conv-FFN) |
| 4 | Oobleck VAE decoder | `vae.ckpt` (decoder subset) | ~0.1 B | build fresh | low (snake, dilated conv, transpose conv, weight-norm fold) |
| 5 | Rectified-flow sampler + CFG | — (algorithm) | — | build fresh | low |

Conditioners in the config are **`mm_unchang`** type = pass-through with `nn.Identity` (dim==output_dim) — **no learned weights** (`conditioners.py:32,212`). All learned conditioning projection lives *inside* the MM-DiT.

## 3. Conventions

- Tensors written `[B, …]`; PyTorch is channels-first for audio (`[B,C,N]`) but the DiT works channels-last (`[B,N,C]`) internally (`mmdit.py:391` permutes in, `:468` permutes out).
- `head_dim = hidden_dim / num_heads = 1024/16 = 64`.
- `modulate(x,shift,scale) = x·(1+scale)+shift` (`transformer_layers.py:20`).
- `SiLU`=`x·σ(x)`; FFNs are **SwiGLU**: `w2(silu(w1·x) · w3·x)` (`blocks.py:382,430`).
- LayerNorms in the DiT are **affine-free** (`elementwise_affine=False`) — pure normalize, modulation supplies scale/shift.
- All Linear layers have bias **unless** noted (SwiGLU `w1/w2/w3` are bias-free; `final conv` of VAE bias-free).

---

## 4. Component 1 — MetaCLIP text tower

`facebook/metaclip-h14-fullcc2.5b`, loaded as a HF `CLIPModel`; only the **text** side is used (`feature_utils_224.py:66-70`). The reference calls a patched `get_text_features` plus the raw `last_hidden_state` to produce **two** outputs (`feature_utils_224.py:21-48`, `encode_text`):

- `metaclip_global_text_features` = `text_projection(pooled_output)` → `[B,1024]` (pooled = EOS-token hidden state).
- `metaclip_text_features` = per-token `last_hidden_state` → `[B,77,1024]`.

Architecture (standard CLIP text transformer, ViT-H/14 config):
- token embedding: vocab **49408** × width **1024**; positional embedding ctx **77** × 1024.
- **24** residual transformer layers, **16** heads, head_dim 64, MLP ratio 4 (GELU, `quick_gelu` for metaclip — **[verify]** quick-gelu vs gelu), **causal** attention mask.
- final `LayerNorm`, then `text_projection` Linear(1024→1024, no bias) applied to the EOS-position hidden state for the global feature.
- Tokenizer: CLIP BPE (vocab+merges), lowercase, BOS/EOS, pad/truncate to 77.

GGML: a small graph — token+pos embed → 24×(LN, causal MHA, LN, GELU-MLP) → final LN; pooled→projection. All native ops. Output **both** the per-token states (pre-projection, post-final-LN per CLIP convention — **[verify]** whether `metaclip_text_features` is pre- or post-final-LN) and the projected pooled vector.

## 5. Component 2 — T5-v1.1-xl encoder

`google/t5-v1_1-xl`, used as a `T5EncoderModel` (`feature_utils_224.py:68-69`). Tokenize with `max_length=77, padding="max_length", truncation=True` (`extract_latents.py`), output `last_hidden_state` → `[B,77,2048]`.

Standard T5 v1.1 encoder: `d_model=2048`, `d_ff=5120` (gated-GELU FFN: `wi_0`/`wi_1`/`wo`), `num_layers=24`, `num_heads=32`, RMSNorm (T5 LayerNorm, no bias, no mean-subtraction), **relative position bias** (shared, bucketed, 32 buckets, only layer-0 attention computes it), no scaling of attention logits. SentencePiece tokenizer (`spiece.model`), vocab 32128.

GGML: **reuse llama.cpp `LLM_ARCH_T5ENCODER`** (`src/models/t5encoder.cpp`) — convert via `convert_hf_to_gguf.py`. Drive it to emit raw encoder `last_hidden_state`. Fallback: a native ggml T5-encoder graph (the relative-bias + gated-GELU are all native). *Open question: confirm libllama emits hidden states identical to HF `last_hidden_state` incl. padding handling.*

---

## 6. Component 3 — MM-DiT ("MMmodule")  ← the core

Config (`thinksound.json` → `model.diffusion.config`): `latent_dim 64, clip_dim 1024, sync_dim 768, text_dim 2048, hidden_dim 1024, depth 21, fused_depth 14, num_heads 16, latent_seq_len 194, clip_seq_len 72, sync_seq_len 216, text_seq_len 77, v2 true, kernel_size 3, mlp_ratio 4.0`. Default `sync_kernel 7`. Flags `add_video / cross_attend / triple_fusion / gated_video / use_inpaint / use_mlp` are all **False** for this config → the text path is the clean subset.

**Block counts** (`mmdit.py:175-185`): `joint_blocks = depth - fused_depth = 7`; `fused_blocks = fused_depth = 14`. In the last joint block (`i==6`) the clip & text sub-blocks are `pre_only` (contribute K/V to joint attention but are not themselves updated).

### 6.1 Submodules & tensors (state_dict keys under the DiT root)

> Root prefix in the ckpt is `model.model.*` (the `MMConditionedDiffusionModelWrapper.model` is the `MMDiTWrapper`, whose `.model` is the `MMmodule`) — **[confirm M0]** exact depth of `model.` nesting.

**Input projections** (`mmdit.py:98-152`, v2 path):
| module | layers | tensors (shapes) |
|---|---|---|
| `audio_input_proj` | ChannelLastConv1d(64→1024,k3,p1) → SiLU → ConvMLP(1024,4096,k3) | `.0.{weight[1024,64,3],bias[1024]}`, `.2.{w1,w3 [2816,1024,3], w2 [1024,2816,3]}` |
| `clip_input_proj` | Linear(1024→1024) → SiLU → ConvMLP(1024,4096,k3) | `.0.{weight[1024,1024],bias}`, `.2.{w1,w2,w3}` |
| `sync_input_proj` | ChannelLastConv1d(768→1024,k7,p3) → SiLU → ConvMLP(1024,4096,k3) | `.0.{weight[1024,768,7],bias}`, `.2.{w1,w2,w3}` |
| `text_input_proj` | Linear(2048→1024) → SiLU → MLP(1024,4096) | `.0.{weight[1024,2048],bias}`, `.2.{w1,w2,w3}` |

ConvMLP/MLP hidden = `multiple_of(256, int(2·4096/3)) = 2816`.

**Conditioning & misc** (`mmdit.py:153-195`):
| tensor | shape | note |
|---|---|---|
| `clip_cond_proj` | Linear 1024→1024 `{weight,bias}` | applied to `clip_f.mean(seq)` |
| `text_cond_proj` | Linear 1024→1024 `{weight,bias}` | applied to `metaclip_global_text_features` |
| `global_cond_mlp` | MLP(1024,4096) `{w1,w2,w3}` | SwiGLU |
| `sync_pos_emb` | `[1,1,8,768]` | added per 8-frame sync segment |
| `t_embed.mlp.0` / `.2` | Linear 1024→1024 ×2 `{weight,bias}` | timestep MLP |
| `final_layer.adaLN_modulation.1` | Linear 1024→2048 `{weight,bias}` | shift,scale |
| `final_layer.conv` | ChannelLastConv1d 1024→64,k7,p3 `{weight,bias}` | output proj |
| `empty_clip_feat` | `[1,1024]` | learned null video-CLIP |
| `empty_sync_feat` | `[1,768]` | learned null sync |
| `empty_string_feat` | `[77,1024]` | learned null CLIP-text |
| `empty_t5_feat` | `[77,2048]` | **zeros**, not trained (`mmdit.py:190,193`) |

`t_embed.freqs`, `latent_rot`, `clip_rot` are **non-persistent buffers** — recompute at load (`mmdit.py:506-511` strips them).

**Per JointBlock** `joint_blocks.{i}` (i=0..6), three `MMDitSingleBlock`s — `latent_block`(k3), `clip_block`(k3), `text_block`(k1):
For a **full** sub-block:
- `attn.qkv` Linear 1024→3072 `{weight,bias}`; `attn.q_norm.weight[64]`, `attn.k_norm.weight[64]` (RMSNorm).
- `linear1`: ChannelLastConv1d 1024→1024,k3 `{weight,bias}` (latent/clip) **or** Linear 1024→1024 (text, k1).
- `ffn`: ConvMLP(1024,4096,k3) `{w1,w2,w3}` (latent/clip) **or** MLP (text). 
- `adaLN_modulation.1` Linear 1024→**6144** `{weight,bias}` (6·dim).
For a **pre_only** sub-block (clip_block & text_block of block 6): only `norm1`(no params), `attn.{qkv,q_norm,k_norm}`, and `adaLN_modulation.1` Linear 1024→**2048** (2·dim). No `linear1/norm2/ffn`.

**Per fused block** `fused_blocks.{i}` (i=0..13), one full `MMDitSingleBlock`(k3): `attn.{qkv,q_norm,k_norm}`, `linear1`(ChannelLastConv1d k3), `ffn`(ConvMLP k3), `adaLN_modulation.1`(1024→6144).

### 6.2 Timestep embedding (`embeddings.py:43-85`, v2)
`frequency_embedding_size=1024, max_period=1`. `freqs[j] = (10000/1)·10000^(−2j/1024)`, j=0..511. `emb(t)=cat[cos(t·freqs), sin(t·freqs)]` → `[B,1024]`. Then `mlp = Linear(1024→1024) → SiLU → Linear(1024→1024)`.

### 6.3 RoPE (`embeddings.py:16-40`) — **highest-risk op**
FLUX-style. `compute_rope_rotations(L, head_dim=64, theta=10000, freq_scaling)`:
`freqs[k]=freq_scaling·theta^(−2k/64)`, k=0..31; `rot[n,k] = [[cos,−sin],[sin,cos]]` with angle `n·freqs[k]`; shape `[1,L,32,2,2]`.
`apply_rope(x,rot)`: view x as pairs of **adjacent** channels `(x_{2k},x_{2k+1})`, rotate each pair → `x'_{2k}=cos·x_{2k}−sin·x_{2k+1}`, `x'_{2k+1}=sin·x_{2k}+cos·x_{2k+1}`. This is **interleaved/GPT-J style**.
- latent stream: `freq_scaling=1.0`, L=194.
- clip stream: `freq_scaling = latent_seq_len/clip_seq_len = 194/72`, L=72.
- text stream: **no RoPE**.
GGML: either `ggml_rope` "normal" mode (interleaved) with per-stream `freq_scale`, **or** bake the `[cos,sin]` tables as constants and apply the rotation with `mul`/`add` + a pair-swap view. **Must be validated numerically in M2.**

### 6.4 Conditioning preprocessing (`mmdit.py:260-318`, cached across the 24 steps)
1. `sync_f.view(B, 27, 8, 768) + sync_pos_emb` → flatten → `[B,216,768]` → `sync_input_proj` → `[B,216,1024]`.
2. `clip_f = clip_input_proj(clip_f)` → `[B,72,1024]`.
3. `text_f_c = text_cond_proj(metaclip_global_text_features)` → `[B,1024]`.
4. pad `text_f`(77×1024) on last dim to 2048 → concat with `t5_features`(77×2048) along **sequence** → `[B,154,2048]` → `text_input_proj` → `[B,154,1024]`.
5. `sync_f`: transpose → `F.interpolate(size=194, mode='nearest-exact')` → `[B,194,1024]`.
6. `clip_f_c = clip_cond_proj(clip_f.mean(seq))` → `[B,1024]`.

### 6.5 Per-step forward (`mmdit.py:320-378` predict_flow)
```
latent      = audio_input_proj(latent)               # [B,194,1024]
global_c    = global_cond_mlp(clip_f_c + text_f_c)   # [B,1024]
global_c    = t_embed(t)[:,None,:] + global_c[:,None,:]   # [B,1,1024]
extended_c  = global_c + sync_f                       # [B,194,1024]  (broadcast add)
for blk in joint_blocks(7):
    latent, clip_f, text_f = blk(latent, clip_f, text_f, global_c, extended_c, latent_rot, clip_rot)
for blk in fused_blocks(14):
    latent = blk(latent, extended_c, latent_rot)
flow = final_layer(latent, extended_c)               # [B,194,64]
```

**JointBlock** (`transformer_layers.py:230-256`): each stream runs `pre_attention` (latent cond=`extended_c` per-token, rot=`latent_rot`; clip cond=`global_c`, rot=`clip_rot`; text cond=`global_c`, rot=None). Concatenate q,k,v over sequence → `[lat(194); clip(72); text(154)]` = 420 tokens → **one** joint attention (16 heads) → split back → each stream `post_attention` (latent always; clip/text only if not `pre_only`).

**MMDitSingleBlock** (`transformer_layers.py:170-198`):
```
pre:  mod = AdaLN(SiLU(c));  (shift_msa,scale_msa[,gate_msa,shift_mlp,scale_mlp,gate_mlp]) = chunk
      x = modulate(norm1(x), shift_msa, scale_msa)
      qkv = qkv(x) → heads → q=q_norm(q), k=k_norm(k) → rope(q),rope(k)
post: x = x + linear1(attn_out)·gate_msa
      x = x + ffn(modulate(norm2(x), shift_mlp, scale_mlp))·gate_mlp
```
Attention = scaled-dot-product, no mask, scale `1/√64` (`transformer_layers.py:33`).

**FinalBlock** (`transformer_layers.py:259-271`): `shift,scale=AdaLN(SiLU(extended_c)).chunk(2); latent=modulate(norm(latent),shift,scale); latent=conv_k7(latent)` → `[B,194,64]`.

### 6.6 Classifier-free guidance (`mmdit.py:380-469`, batch_cfg)
At inference `cfg_scale=5`, `cfg_dropout_prob=0`. The wrapper **doubles** the batch: half 1 = real conditions, half 2 = **null** conditions = learned empties (`empty_clip/sync/string` expanded, `empty_t5`=zeros, `metaclip_global`=zeros) (`mmdit.py:436-453`). Run the network on `[2B,…]`, then
`flow = uncond + (cond − uncond)·cfg_scale` (`mmdit.py:460-461`), `scale_phi=0` so no std-rescale. For the **text-only** path the "real" clip/sync are already the empties, so CFG effectively guides on the text/T5 conditioning.

---

## 7. Component 4 — Oobleck VAE decoder

Config: `out_channels 2, channels 128, c_mults [1,2,4,8,16], strides [2,4,4,8,8], latent_dim 64, use_snake true, final_tanh false` (`thinksound.json` pretransform). Decoder construction (`autoencoders.py:150-191`) prepends `c_mults=[1]+[…]=[1,1,2,4,8,16]`, depth 6.

Layer order (`x` is `[B,64,194]` channels-first):
1. `WNConv1d(64 → 2048, k7, p3)` (`16·128=2048`).
2. **5× DecoderBlock**, i=5→1: `in=c_mults[i]·128, out=c_mults[i-1]·128, stride=strides[i-1]` →
   - `(2048→1024, s8), (1024→512, s8), (512→256, s4), (256→128, s4), (128→128, s2)`.
   - Each DecoderBlock (`autoencoders.py:83-114`): `SnakeBeta(in) → WNConvTranspose1d(in→out, k=2·stride, stride, p=ceil(stride/2)) → ResidualUnit(d=1) → ResidualUnit(d=3) → ResidualUnit(d=9)`.
3. `SnakeBeta(128) → WNConv1d(128 → 2, k7, p3, bias=False) → Identity` (final_tanh false).

Total upsample = 8·8·4·4·2 = **2048** → output `[B,2,397312]` (194·2048). 

**ResidualUnit** (`autoencoders.py:39-62`): `x + [SnakeBeta(in) → WNConv1d(in→out,k7,dilation=d,p=d·3) → SnakeBeta(out) → WNConv1d(out→out,k1)]`. (Decoder uses in==out.)

**SnakeBeta** (`blocks.py:301-339`, alpha_logscale=True): `α=exp(alpha_param), β=exp(beta_param)`, then `y = x + (1/(β+1e-9))·sin²(α·x)`, with `alpha`/`beta` per-channel `[C]`. Tensor keys `…alpha`, `…beta`.

**Weight-norm**: every `WNConv1d`/`WNConvTranspose1d` stores `weight_g`+`weight_v` → fold to `weight` at conversion (see `GGUF_CONVERSION.md §3`).

**Bottleneck / scaling**: VAE bottleneck `decode` is **identity** (`bottleneck.py:79-80`); `AutoencoderPretransform.decode` multiplies by `scale=1.0` (default, not overridden) (`pretransforms.py:63-64`). **No latent scaling** — the DiT's 64-ch latent feeds the decoder directly.

**Post-decode** (`predict.py:predict_step`): `audio = (wav / max(|wav|)).clamp(-1,1) · 32767 → int16`, written as 44.1 kHz stereo WAV.

GGML: all native — `conv_1d` (with dilation), `conv_transpose_1d` (CUDA kernel present), `group_norm(1,C)` inside residual units (`autoencoders.py` uses `GroupNorm(1,·)` — **[verify]**: oobleck `ResidualUnit` here has *no* GroupNorm; only `ResConvBlock` in `blocks.py` does. The oobleck residual unit is snake+conv only), snake via `sin/sqr/exp/div/add`.

> Note: the oobleck `ResidualUnit`/`DecoderBlock` use **no normalization** — just SnakeBeta + (weight-normed) convs. (GroupNorm appears only in the unused `ResConvBlock`.) Confirm against the loaded keys in M1.

## 8. Component 5 — Rectified-flow sampler + CFG

`sample_discrete_euler` (`sampling.py:24-45`), `steps=24, sigma_max=1`:
```
x = randn[1, 64, 194]                      # initial noise
t = linspace(1.0, 0.0, steps+1)            # 25 points
for (t_curr, t_prev) in zip(t[:-1], t[1:]):
    v  = model_cfg(x, t_curr·ones(B))      # MM-DiT forward with CFG (§6.6)
    x += (t_prev − t_curr) · v             # dt < 0; integrate noise→data
return x                                   # final latent
```
`model_cfg` is the full §6 forward incl. the doubled-batch CFG combine. Output latent → §7 decoder.

RNG: only the initial `randn` is stochastic → seed once to match PyTorch (`seed_everything`). The loop is deterministic.

---

## 9. Duration scaling

`predict.py` recomputes per duration `D` (s): `latent_seq_len = round(44100/64/32·D)`, `clip_seq_len = 8·D`, `sync_seq_len = 24·D`; `sample_size = D·44100`. RoPE tables and `sync` reshape (`//8`) depend on these → recompute on each run. (9 s → 194/72/216.) Text seq stays 77.

## 10. GGML implementation notes (non-native / careful ops)

| Op | Status in `/devel/tools/llama.cpp/ggml` | Strategy |
|---|---|---|
| **FLUX interleaved RoPE w/ per-stream freq_scale** | `ggml_rope` (normal & neox), `freq_scale` arg | Try normal mode + `freq_scale`; else precompute `[cos,sin]` consts + manual pair-rotate (`view`+`mul`+`add`). **Validate M2.** |
| **SnakeBeta** | `sin`,`sqr`,`exp`,`div`,`add`,`scale` | compose `x + (1/exp(β))·sin²(exp(α)·x)`; α,β broadcast over time |
| **Dilated Conv1d** | `ggml_conv_1d(…, d0)` has dilation | direct |
| **ConvTranspose1d** | `ggml_conv_transpose_1d` (+CUDA kernel) | check `output_padding` (`ceil(stride/2)`); validate length parity M1 |
| **ChannelLastConv1d** | conv expects a fixed layout | transpose `[B,N,C]↔[B,C,N]` around `ggml_conv_1d` |
| **Joint attention (concat streams)** | `ggml_concat`, attention | concat q/k/v along seq, single SDPA, slice back |
| **adaLN per-token modulation** | `mul`,`add`,broadcast | `extended_c` is per-token `[B,194,1024]`; AdaLN→`[B,194,6144]`; broadcast-free |
| **affine-free LayerNorm** | `ggml_norm` | no weight/bias |
| **RMSNorm (QK + T5)** | `ggml_rms_norm` | with learned scale (QK head_dim 64) |
| **nearest-exact interpolate** | `ggml_upscale`/interpolate | verify `nearest-exact` vs `nearest` alignment for sync→194 |
| **Batch-CFG** | graph-level | build `[2B]` batch, null half from empties, combine `uncond+(cond-uncond)·5` |
| **timestep embed (max_period=1)** | `sin`,`cos` | bespoke freqs (not `ggml_timestep_embedding`'s default scaling) |

All ops have a native ggml/CUDA path; **no kernel needs to be written**. The only genuine *numerical-parity* risks are RoPE layout (§6.3) and `nearest-exact` interpolation alignment — both pinned by the M1/M2 golden-tensor tests.

## 11. Open questions to verify against weights / reference (golden dumps)

1. **RoPE**: `ggml_rope` normal vs the FLUX pair layout, and the clip-stream `freq_scaling=194/72`. (M2)
2. **Checkpoint nesting**: exact prefix (`model.model.*`?), and whether `thinksound_light.ckpt` is the EMA/inference copy directly `load_state_dict`-able. Read `unwrap.py`. (M0)
3. **MetaCLIP details**: quick-gelu vs gelu; whether `metaclip_text_features` is pre- or post-final-LN; the exact pooled/EOS index. (M4)
4. **T5 reuse**: libllama `T5ENCODER` reproducing HF `last_hidden_state` incl. 77-pad. (M5)
5. **`nearest-exact`** interpolation alignment for sync→latent length. (M2)
6. **VAE residual-unit normalization**: confirm oobleck residual units have no GroupNorm (only snake+conv) from the loaded key set. (M1)
7. **ConvTranspose output length** parity (output_padding) at each stride. (M1)
8. Duration generality (non-9 s) for seq-len/RoPE recompute. (M6)

---
### Appendix A — exact shapes for B=1, 9 s (post-CFG B=2)
`x[1,64,194]`→permute`[1,194,64]`; clip`[2,72,1024]`; sync`[2,216,768]`→`[2,194,1024]`; text in `[2,77,1024]`+t5`[2,77,2048]`→concat`[2,154,2048]`→`[2,154,1024]`; global_text`[2,1024]`; joint-attn tokens `194+72+154=420`; flow`[2,194,64]`→cfg→`[1,194,64]`→permute`[1,64,194]`→decode→`[1,2,397312]`.
