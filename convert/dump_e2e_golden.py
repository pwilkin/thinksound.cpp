#!/usr/bin/env python3
"""Ground-truth end-to-end golden for a real text prompt:
load the reference text features (from golden_text.gguf), run the reference DiT sampler
(text-only: video=empties) + VAE decode -> reference latent + wav. Fixed noise seed.
"""
import sys, types, argparse, json
sys.modules['k_diffusion'] = types.ModuleType('k_diffusion')
import torch, numpy as np
from gguf import GGUFWriter
import gguf as gguflib
REPO = "/tmp/ThinkSound"; sys.path.insert(0, REPO)
from ThinkSound.models.factory import create_model_from_config
from ThinkSound.inference.sampling import sample_discrete_euler

def read_gguf_tensor(path, name):
    r = gguflib.GGUFReader(path)
    for t in r.tensors:
        if t.name == name:
            return torch.from_numpy(np.array(t.data)).reshape(list(reversed(t.shape))).float()
    raise KeyError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/thinksound_light.ckpt")
    ap.add_argument("--cfg",  default=f"{REPO}/ThinkSound/configs/model_configs/thinksound.json")
    ap.add_argument("--text", default="/media/ilintar/D_SSD/thinksound/golden/golden_text.gguf")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/golden/golden_e2e.gguf")
    ap.add_argument("--wav",  default="/media/ilintar/D_SSD/thinksound/golden/golden_e2e_ref.wav")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    cfg = json.load(open(a.cfg)); D = 9.0
    cfg["sample_size"] = D * cfg["sample_rate"]
    cfg["model"]["diffusion"]["config"]["sync_seq_len"]   = 24 * int(D)
    cfg["model"]["diffusion"]["config"]["clip_seq_len"]   = 8 * int(D)
    cfg["model"]["diffusion"]["config"]["latent_seq_len"] = round(44100/64/32*D)
    model = create_model_from_config(cfg).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False), strict=False)
    dit = model.model.model

    lat_len, clip_len, sync_len = round(44100/64/32*D), 8*int(D), 24*int(D)
    # real text features [from golden_text]: stored ggml ne=[feat,seq] -> torch [seq,feat]
    t5 = read_gguf_tensor(a.text, "t5_features").reshape(77, 2048).unsqueeze(0)
    mt = read_gguf_tensor(a.text, "metaclip_text_features").reshape(77, 1024).unsqueeze(0)
    mg = read_gguf_tensor(a.text, "metaclip_global_text_features").reshape(1024).unsqueeze(0)
    cond = {
        "metaclip_features": dit.empty_clip_feat.detach().unsqueeze(0).expand(1, clip_len, -1).contiguous(),
        "sync_features":     dit.empty_sync_feat.detach().unsqueeze(0).expand(1, sync_len, -1).contiguous(),
        "metaclip_text_features": mt, "metaclip_global_text_features": mg, "t5_features": t5,
    }
    ci = model.get_conditioning_inputs(cond)
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn(1, model.io_channels, lat_len, generator=g, dtype=torch.float32)
    with torch.no_grad():
        latent = sample_discrete_euler(model.model, noise, 24, cfg_scale=5.0, batch_cfg=True, **ci)
        wav = model.pretransform.decode(latent)
    print("latent", tuple(latent.shape), "wav", tuple(wav.shape))

    w = GGUFWriter(a.out, arch="thinksound-e2e-golden")
    w.add_tensor("noise", noise.numpy()); w.add_tensor("latent", latent.numpy()); w.add_tensor("wav_ref", wav.float().numpy())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.out)
    import wave
    aud = (wav[0] / wav.abs().max()).clamp(-1,1).numpy()
    pcm = (aud.T * 32767.0).round().astype("<i2")
    with wave.open(a.wav, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(44100); wf.writeframes(pcm.tobytes())
    print("wrote", a.wav)

if __name__ == "__main__":
    main()
