#pragma once
#include "common/ts_model.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// CLIP byte-level BPE tokenizer (matches HF CLIPTokenizer used by MetaCLIP).
struct ts_clip_tokenizer {
    std::unordered_map<std::string, int32_t> vocab;        // token -> id
    std::unordered_map<std::string, int32_t> merge_rank;   // "a b" -> rank
    std::string byte2u[256];                               // GPT-2 byte->unicode
    int32_t bos_id = 49406, eos_id = 49407;
    bool load(const std::string & gguf_path);
    // -> ids padded/truncated to `ctx` (default 77) with eos, bos prefixed.
    std::vector<int32_t> encode(const std::string & text, int ctx = 77) const;
};

// T5 SentencePiece (unigram) tokenizer.
struct ts_t5_tokenizer {
    std::vector<std::string> pieces;
    std::unordered_map<std::string, int32_t> piece2id;
    std::vector<float> scores;
    int32_t unk_id = 2, eos_id = 1, pad_id = 0;
    bool load(const std::string & gguf_path);
    // -> ids with </s> appended, padded/truncated to `ctx` (default 77) with pad.
    std::vector<int32_t> encode(const std::string & text, int ctx = 77) const;
};
