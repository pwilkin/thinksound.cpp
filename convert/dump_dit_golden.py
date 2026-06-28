#!/usr/bin/env python3
"""Golden reference for the MM-DiT + sampler (text-only path).
Builds the full ThinkSound model, loads thinksound_light.ckpt, uses FIXED synthetic
conditioning (video features -> learned empties; text/t5 -> seeded random), runs:
  - one forward (step-0 velocity) for incremental parity,
  - the full 24-step rectified-flow Euler sampler (cfg=5),
  - VAE decode,
and dumps every boundary tensor to golden_dit.gguf for the C++ parity tests.
"""
import sys, types, argparse, json
sys.modules['k_diffusion'] = types.ModuleType('k_diffusion')
import torch, numpy as np
from gguf import GGUFWriter

REPO = "/tmp/ThinkSound"
sys.path.insert(0, REPO)
from ThinkSound.models.factory import create_model_from_config
from ThinkSound.inference.sampling import sample_discrete_euler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/thinksound_light.ckpt")
    ap.add_argument("--vae",  default="/media/ilintar/D_SSD/thinksound/ckpts/vae.ckpt")
    ap.add_argument("--cfg",  default=f"{REPO}/ThinkSound/configs/model_configs/thinksound.json")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/golden/golden_dit.gguf")
    ap.add_argument("--wav",  default="/media/ilintar/D_SSD/thinksound/golden/golden_dit_ref.wav")
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = json.load(open(args.cfg))
    D = args.duration
    cfg["sample_size"] = D * cfg["sample_rate"]
    cfg["model"]["diffusion"]["config"]["sync_seq_len"]   = 24 * int(D)
    cfg["model"]["diffusion"]["config"]["clip_seq_len"]   = 8 * int(D)
    cfg["model"]["diffusion"]["config"]["latent_seq_len"] = round(44100 / 64 / 32 * D)
    model = create_model_from_config(cfg).eval()
    miss, unexp = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False), strict=False)
    print("DiT load: missing", len(miss), "unexpected", len(unexp))

    dev = "cpu"
    lat_len  = round(44100 / 64 / 32 * D)        # 194
    clip_len = 8 * int(D)                         # 72
    sync_len = 24 * int(D)                        # 216
    g = torch.Generator().manual_seed(args.seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)

    dit = model.model.model            # MMmodule (has empty_* params)
    empty_clip = dit.empty_clip_feat.detach()     # [1,1024]
    empty_sync = dit.empty_sync_feat.detach()     # [1,768]

    cond = {
        # video features -> learned empties (text-only path, matches predict.py)
        "metaclip_features":  empty_clip.unsqueeze(0).expand(1, clip_len, -1).contiguous(),
        "sync_features":      empty_sync.unsqueeze(0).expand(1, sync_len, -1).contiguous(),
        # text encoder outputs -> seeded random stand-ins
        "metaclip_text_features":        rnd(1, 77, 1024),
        "metaclip_global_text_features": rnd(1, 1024),
        "t5_features":                   rnd(1, 77, 2048),
    }
    cond_inputs = model.get_conditioning_inputs(cond)

    noise = rnd(1, model.io_channels, lat_len)    # [1,64,194]

    with torch.no_grad():
        t0 = torch.ones(1)                        # step-0 t = sigma_max = 1
        flow0 = model.model(noise, t0, cfg_scale=args.cfg_scale, batch_cfg=True, **cond_inputs)
        latent = sample_discrete_euler(model.model, noise, args.steps,
                                       cfg_scale=args.cfg_scale, batch_cfg=True, **cond_inputs)
        wav = model.pretransform.decode(latent)

    print("flow0", tuple(flow0.shape), "latent", tuple(latent.shape), "wav", tuple(wav.shape))

    w = GGUFWriter(args.out, arch="thinksound-golden-dit")
    def put(name, t): w.add_tensor(name, t.detach().to(torch.float32).contiguous().numpy())
    put("noise", noise)
    put("clip_f", cond_inputs["clip_f"]); put("sync_f", cond_inputs["sync_f"])
    put("text_f", cond_inputs["text_f"]); put("t5_features", cond_inputs["t5_features"])
    put("metaclip_global_text_features", cond_inputs["metaclip_global_text_features"])
    put("flow0", flow0); put("latent", latent); put("wav_ref", wav)
    w.add_float32("gen.cfg_scale", args.cfg_scale); w.add_uint32("gen.steps", args.steps)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", args.out)

    import wave
    a = (wav[0] / wav.abs().max()).clamp(-1, 1).numpy()
    pcm = (a.T * 32767.0).round().astype("<i2")
    with wave.open(args.wav, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(44100); wf.writeframes(pcm.tobytes())
    print("wrote", args.wav)

if __name__ == "__main__":
    main()
