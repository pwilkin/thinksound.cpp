#!/usr/bin/env python3
"""Golden for the text encoders: tokenize a sample (caption, cot) with the reference
tokenizers, run T5-v1.1-xl encoder + MetaCLIP text tower, dump token IDs + features.
Also prints the encoder state_dict key layout for the GGUF converters.
"""
import sys, types, argparse
sys.modules['k_diffusion'] = types.ModuleType('k_diffusion')
import torch
from gguf import GGUFWriter
from transformers import T5EncoderModel, AutoTokenizer, AutoModel, AutoProcessor

T5_PATH   = "/media/ilintar/D_SSD/thinksound/hf/t5-v1_1-xl"
CLIP_PATH = "/media/ilintar/D_SSD/thinksound/hf/metaclip-h14"

def patch_clip(clip_model):
    def f(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.text_projection(out[1]), out[0]   # (global[B,1024], per-token[B,77,1024])
    clip_model.get_text_features = f.__get__(clip_model)
    return clip_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", default="a dog barking")
    ap.add_argument("--cot", default="A dog barks several times, sharp and clear, in a quiet room.")
    ap.add_argument("--out", default="/media/ilintar/D_SSD/thinksound/golden/golden_text.gguf")
    a = ap.parse_args()

    t5tok = AutoTokenizer.from_pretrained(T5_PATH)
    t5 = T5EncoderModel.from_pretrained(T5_PATH, torch_dtype=torch.float32).eval()
    ti = t5tok([a.cot], truncation=True, max_length=77, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        t5_features = t5(**ti).last_hidden_state.float()      # [1,77,2048]

    clip = patch_clip(AutoModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float32).eval())
    proc = AutoProcessor.from_pretrained(CLIP_PATH)
    ci = proc(text=[a.caption], truncation=True, max_length=77, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        g_feat, t_feat = clip.get_text_features(input_ids=ci["input_ids"], attention_mask=ci.get("attention_mask"))

    print("t5_ids", ti["input_ids"].shape, ti["input_ids"][0][:12].tolist())
    print("clip_ids", ci["input_ids"].shape, ci["input_ids"][0][:12].tolist())
    print("t5_features", tuple(t5_features.shape), "metaclip_text", tuple(t_feat.shape), "global", tuple(g_feat.shape))

    # state_dict key samples for converters
    print("\n[T5 keys]");  [print("  ", k, tuple(v.shape)) for k, v in list(t5.state_dict().items())[:8]]
    print("[CLIP text keys]"); [print("  ", k, tuple(v.shape)) for k, v in clip.state_dict().items() if k.startswith("text_model.encoder.layers.0.") or k.startswith("text_model.embeddings") or k=="text_projection.weight"][:14]

    w = GGUFWriter(a.out, arch="thinksound-text-golden")
    w.add_tensor("t5_input_ids",   ti["input_ids"].to(torch.int32).numpy())
    w.add_tensor("clip_input_ids", ci["input_ids"].to(torch.int32).numpy())
    w.add_tensor("t5_features", t5_features.numpy())
    w.add_tensor("metaclip_text_features", t_feat.float().numpy())
    w.add_tensor("metaclip_global_text_features", g_feat.float().numpy())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.out)

if __name__ == "__main__":
    main()
