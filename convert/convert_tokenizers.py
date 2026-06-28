#!/usr/bin/env python3
"""Extract CLIP-BPE and T5-SentencePiece tokenizer data into small GGUF files
that the C++ tokenizers load. Keeps the big model GGUFs untouched."""
import argparse, json
from gguf import GGUFWriter

def clip_tok(path, out):
    vocab = json.load(open(f"{path}/vocab.json"))           # token -> id
    id2tok = [None] * (max(vocab.values()) + 1)
    for t, i in vocab.items(): id2tok[i] = t
    id2tok = [t if t is not None else "" for t in id2tok]
    merges = []
    for ln in open(f"{path}/merges.txt", encoding="utf-8"):
        ln = ln.rstrip("\n")
        if ln.startswith("#") or not ln: continue
        merges.append(ln)                                    # "a b"
    w = GGUFWriter(out, arch="thinksound-clip-tok")
    w.add_uint32("clip.bos_id", 49406); w.add_uint32("clip.eos_id", 49407)
    w.add_array("clip.vocab", id2tok)
    w.add_array("clip.merges", merges)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: vocab {len(id2tok)}, merges {len(merges)}")

def t5_tok(path, out):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=f"{path}/spiece.model")
    n = sp.vocab_size()
    pieces = [sp.id_to_piece(i) for i in range(n)]
    scores = [float(sp.get_score(i)) for i in range(n)]
    w = GGUFWriter(out, arch="thinksound-t5-tok")
    w.add_uint32("t5.unk_id", sp.unk_id()); w.add_uint32("t5.eos_id", sp.eos_id())
    w.add_uint32("t5.pad_id", sp.pad_id()); w.add_uint32("t5.vocab", n)
    w.add_array("t5.pieces", pieces)
    w.add_array("t5.scores", scores)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: pieces {n}, unk={sp.unk_id()} eos={sp.eos_id()} pad={sp.pad_id()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/media/ilintar/D_SSD/thinksound/hf/metaclip-h14")
    ap.add_argument("--t5", default="/media/ilintar/D_SSD/thinksound/hf/t5-v1_1-xl")
    ap.add_argument("--outdir", default="/media/ilintar/D_SSD/thinksound/gguf")
    a = ap.parse_args()
    clip_tok(a.clip, f"{a.outdir}/clip-tokenizer.gguf")
    t5_tok(a.t5, f"{a.outdir}/t5-tokenizer.gguf")
