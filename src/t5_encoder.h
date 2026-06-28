#pragma once
#include "common/ts_model.h"
#include <vector>
#include <string>

// T5-v1.1-xl encoder (RMSNorm, relative-position bias, gated-gelu FFN).
struct ts_t5 {
    ts_model model;
    ggml_backend_t backend = nullptr;
    int d_model = 2048, d_ff = 5120, NL = 24, NH = 32, hd = 64;
    int rel_buckets = 32, rel_max_dist = 128;
    float eps = 1e-6f;
    ~ts_t5();
    bool load(const std::string & p);
    // ids: n token ids -> last_hidden_state [d_model*n] (ne[0]=d_model, ne[1]=n)
    void encode(const int32_t * ids, int n, std::vector<float> & out);
};
