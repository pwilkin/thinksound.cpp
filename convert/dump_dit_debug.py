#!/usr/bin/env python3
"""Fast single-forward DiT debug golden: cfg_scale=1.0 (no CFG), real seeded conditioning,
with hooks capturing stage outputs so the C++ DiT can be verified incrementally.
Reuses the SAME seed/inputs as dump_dit_golden.py (so inputs match golden_dit.gguf).
"""
import sys, types, argparse, json
sys.modules['k_diffusion'] = types.ModuleType('k_diffusion')
import torch
from gguf import GGUFWriter
REPO = "/tmp/ThinkSound"; sys.path.insert(0, REPO)
from ThinkSound.models.factory import create_model_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/thinksound_light.ckpt")
    ap.add_argument("--cfg",  default=f"{REPO}/ThinkSound/configs/model_configs/thinksound.json")
    ap.add_argument("--out",  default="/media/ilintar/D_SSD/thinksound/golden/golden_dit_dbg.gguf")
    ap.add_argument("--seed", type=int, default=1234)
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
    g = torch.Generator().manual_seed(a.seed); rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    cond = {
        "metaclip_features": dit.empty_clip_feat.detach().unsqueeze(0).expand(1, clip_len, -1).contiguous(),
        "sync_features":     dit.empty_sync_feat.detach().unsqueeze(0).expand(1, sync_len, -1).contiguous(),
        "metaclip_text_features":        rnd(1, 77, 1024),
        "metaclip_global_text_features": rnd(1, 1024),
        "t5_features":                   rnd(1, 77, 2048),
    }
    ci = model.get_conditioning_inputs(cond)
    noise = rnd(1, model.io_channels, lat_len)

    caps = {}
    def cap(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[name] = o.detach().float().contiguous()
        return hook
    dit.audio_input_proj.register_forward_hook(cap("x_in"))
    dit.t_embed.register_forward_hook(cap("t_emb"))
    dit.joint_blocks[-1].register_forward_hook(cap("after_joint"))
    dit.fused_blocks[-1].register_forward_hook(cap("after_fused"))
    dit.final_layer.register_forward_hook(cap("flow_cl"))   # [B,N,latent] channels-last

    with torch.no_grad():
        flow = model.model(noise, torch.ones(1), cfg_scale=1.0, batch_cfg=True, **ci)  # cfg=1 -> single fwd

    print({k: tuple(v.shape) for k, v in caps.items()}, "flow", tuple(flow.shape))
    w = GGUFWriter(a.out, arch="thinksound-dit-dbg")
    def put(n, t): w.add_tensor(n, t.detach().float().contiguous().numpy())
    for k, v in caps.items(): put(k, v)
    put("flow", flow)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.out)

if __name__ == "__main__":
    main()
