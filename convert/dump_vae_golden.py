#!/usr/bin/env python3
"""Golden reference for the VAE decoder: build the real ThinkSound oobleck VAE,
decode a fixed latent, and save {latent, wav_ref} to a GGUF for the C++ parity test.
Also writes a listenable wav.
"""
import sys, types, argparse, json
sys.modules['k_diffusion'] = types.ModuleType('k_diffusion')  # avoid broken transitive dep
import torch, numpy as np
from gguf import GGUFWriter

REPO = "/tmp/ThinkSound"
sys.path.insert(0, REPO)
from ThinkSound.models.factory import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/vae.ckpt")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/golden/golden_vae.gguf")
    ap.add_argument("--wav",  default="/media/ilintar/D_SSD/thinksound/golden/golden_vae_ref.wav")
    ap.add_argument("--seqlen", type=int, default=194)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = json.load(open(f"{REPO}/ThinkSound/configs/model_configs/stable_audio_2_0_vae.json"))
    m = create_model_from_config(cfg).eval()
    sd = load_ckpt_state_dict(args.ckpt, prefix="autoencoder.")
    m.load_state_dict(sd, strict=True)

    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, 64, args.seqlen, generator=g, dtype=torch.float32)
    with torch.no_grad():
        wav = m.decode(latent)            # [1, 2, seqlen*2048]
    wav = wav.to(torch.float32)
    print("latent", tuple(latent.shape), "wav", tuple(wav.shape),
          "wav range", float(wav.min()), float(wav.max()))

    w = GGUFWriter(args.out, arch="thinksound-golden")
    w.add_tensor("latent",  latent.numpy())
    w.add_tensor("wav_ref", wav.numpy())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", args.out)

    # listenable reference (normalize like predict.py), written via stdlib wave
    import wave
    a = (wav[0] / wav.abs().max()).clamp(-1, 1).numpy()          # [2, N]
    pcm = (a.T * 32767.0).round().astype("<i2")                  # [N, 2] interleaved
    with wave.open(args.wav, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(44100)
        wf.writeframes(pcm.tobytes())
    print("wrote", args.wav)

if __name__ == "__main__":
    main()
