#pragma once
#include "common/ts_model.h"
#include <vector>
#include <string>

// MetaCLIP text tower (pre-LN causal transformer).
struct ts_clip_text {
    ts_model model;
    ggml_backend_t backend = nullptr;
    int H = 1024, NL = 24, NH = 16, hd = 64, ctx = 77, eos_id = 49407;
    ~ts_clip_text();
    bool load(const std::string & p);
    // ids: n token ids. Fills per_token [H*n] (ne[0]=H,ne[1]=n) and global [H].
    void encode(const int32_t * ids, int n, std::vector<float> & per_token, std::vector<float> & global);
};
