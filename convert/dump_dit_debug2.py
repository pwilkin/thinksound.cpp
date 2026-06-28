#!/usr/bin/env python3
"""Staged DiT debug for the E2E inputs (reference text features + e2e noise), cfg=1 single forward.
Emits inputs file + stages file in the SAME schema as golden_dit.gguf / golden_dit_dbg.gguf
so ts-dit_test can localize where my DiT diverges for real (structured) features."""
import sys, types, argparse, json
sys.modules['k_diffusion'] = types.ModuleType('k_diffusion')
import torch, numpy as np
from gguf import GGUFWriter, GGUFReader
REPO = "/tmp/ThinkSound"; sys.path.insert(0, REPO)
from ThinkSound.models.factory import create_model_from_config

def rd(path, name):
    r = GGUFReader(path)
    for t in r.tensors:
        if t.name == name:
            return torch.from_numpy(np.array(t.data)).reshape(list(reversed(t.shape))).float()
    raise KeyError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/ilintar/D_SSD/thinksound/ckpts/thinksound_light.ckpt")
    ap.add_argument("--cfg",  default=f"{REPO}/ThinkSound/configs/model_configs/thinksound.json")
    ap.add_argument("--text", default="/media/ilintar/D_SSD/thinksound/golden/golden_text.gguf")
    ap.add_argument("--e2e",  default="/media/ilintar/D_SSD/thinksound/golden/golden_e2e.gguf")
    ap.add_argument("--inp",  default="/media/ilintar/D_SSD/thinksound/golden/golden_e2e_inputs.gguf")
    ap.add_argument("--dbg",  default="/media/ilintar/D_SSD/thinksound/golden/golden_e2e_dbg.gguf")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg)); D = 9.0
    cfg["sample_size"] = D*cfg["sample_rate"]
    cfg["model"]["diffusion"]["config"]["sync_seq_len"]=24*int(D)
    cfg["model"]["diffusion"]["config"]["clip_seq_len"]=8*int(D)
    cfg["model"]["diffusion"]["config"]["latent_seq_len"]=round(44100/64/32*D)
    model = create_model_from_config(cfg).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False), strict=False)
    dit = model.model.model
    cl, sl = 8*int(D), 24*int(D)
    t5 = rd(a.text, "t5_features").reshape(77,2048).unsqueeze(0)
    mt = rd(a.text, "metaclip_text_features").reshape(77,1024).unsqueeze(0)
    mg = rd(a.text, "metaclip_global_text_features").reshape(1024).unsqueeze(0)
    noise = rd(a.e2e, "noise").reshape(1,64,194)
    cond = {"metaclip_features": dit.empty_clip_feat.detach().unsqueeze(0).expand(1,cl,-1).contiguous(),
            "sync_features": dit.empty_sync_feat.detach().unsqueeze(0).expand(1,sl,-1).contiguous(),
            "metaclip_text_features": mt, "metaclip_global_text_features": mg, "t5_features": t5}
    ci = model.get_conditioning_inputs(cond)

    caps = {}
    def cap(n):
        def hook(m,i,o): caps[n] = (o[0] if isinstance(o,tuple) else o).detach().float().contiguous()
        return hook
    # patch the joint attention to capture block-0 (first call) q/k/v/output
    import ThinkSound.models.transformer_layers as TL
    _orig_attn = TL.attention
    _seen = [0]
    def patched(q, k, v):
        out = _orig_attn(q, k, v)
        if _seen[0] == 0:
            caps["a0_q"] = q[0].permute(1,0,2).contiguous().float()  # [h,n,d]->[n,h,d]
            caps["a0_k"] = k[0].permute(1,0,2).contiguous().float()
            caps["a0_v"] = v[0].permute(1,0,2).contiguous().float()
            caps["a0_out"] = out[0].contiguous().float()             # [n, h*d]
            _seen[0] = 1
        return out
    TL.attention = patched
    _mod_seen = [0]
    _orig_mod = TL.modulate
    def patched_mod(x, shift, scale):
        out = _orig_mod(x, shift, scale)
        _mod_seen[0] += 1
        if _mod_seen[0] == 3:  # block0: latent(1), clip(2), text(3); x is [b,n,H]
            caps["text_norm1"] = x[0].contiguous().float()   # [n,H] C-flat = H-fastest = my [H,n]
            caps["text_xn"] = out[0].contiguous().float()
        return out
    TL.modulate = patched_mod
    dit.audio_input_proj.register_forward_hook(cap("x_in"))
    dit.t_embed.register_forward_hook(cap("t_emb"))
    dit.joint_blocks[-1].register_forward_hook(cap("after_joint"))
    dit.fused_blocks[-1].register_forward_hook(cap("after_fused"))
    dit.final_layer.register_forward_hook(cap("flow_cl"))
    # capture preprocessed conditioning entering joint block 0: (latent, clip_f, text_f, global_c, extended_c, ...)
    def pre0(mod, args):
        caps["clip_pp"]=args[1].detach().float().contiguous(); caps["text_pp"]=args[2].detach().float().contiguous()
        caps["global_c"]=args[3].detach().float().contiguous(); caps["extended_c"]=args[4].detach().float().contiguous()
    dit.joint_blocks[0].register_forward_pre_hook(pre0)
    dit.clip_cond_proj.register_forward_hook(cap("clip_fc"))
    dit.text_cond_proj.register_forward_hook(cap("text_fc"))
    dit.global_cond_mlp.register_forward_hook(cap("gcmlp"))
    def jb0(mod, inp, out):
        caps["jb0_lat"]=out[0].detach().float().contiguous(); caps["jb0_clip"]=out[1].detach().float().contiguous(); caps["jb0_text"]=out[2].detach().float().contiguous()
    dit.joint_blocks[0].register_forward_hook(jb0)
    with torch.no_grad():
        flow = model.model(noise, torch.ones(1), cfg_scale=1.0, batch_cfg=True, **ci)
    print({k:tuple(v.shape) for k,v in caps.items()})

    wi = GGUFWriter(a.inp, arch="dbg-inp")
    def p(w,n,t): w.add_tensor(n, t.detach().float().contiguous().numpy())
    p(wi,"noise",noise); p(wi,"clip_f",ci["clip_f"]); p(wi,"sync_f",ci["sync_f"])
    p(wi,"text_f",ci["text_f"]); p(wi,"t5_features",ci["t5_features"]); p(wi,"metaclip_global_text_features",ci["metaclip_global_text_features"])
    wi.write_header_to_file(); wi.write_kv_data_to_file(); wi.write_tensors_to_file(); wi.close()
    wd = GGUFWriter(a.dbg, arch="dbg")
    for k,v in caps.items(): p(wd,k,v)
    p(wd,"flow",flow)
    wd.write_header_to_file(); wd.write_kv_data_to_file(); wd.write_tensors_to_file(); wd.close()
    print("wrote", a.inp, a.dbg)

if __name__ == "__main__":
    main()
