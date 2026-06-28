#pragma once
#include "common/ts_model.h"
#include <vector>
#include <string>
#include <map>

// MM-DiT (MMAudio-derived) flow-matching transformer, text-only path, B=1.
// Canonical hidden layout: [hidden, seq] (ne[0]=feature).
struct ts_dit {
    ts_model model;
    ggml_backend_t backend = nullptr;
    int hidden = 1024, n_head = 16, head_dim = 64, depth = 21, fused_depth = 14;
    int latent_dim = 64, text_seq = 77;
    float rope_theta = 10000.f, t_max_period = 1.f;

    ~ts_dit();
    bool load(const std::string & gguf_path);

    // Conditioning inputs (ggml layouts as in golden_dit.gguf):
    //   clip_f [1024,72], sync_f [768,216], text_f [1024,77], t5 [2048,77], global_text [1024]
    // latent [latent_dim, T] (ne[0]=latent_dim). t in [0,1].
    // Returns flow [latent_dim, T]. If dbg != null, fills named stage tensors (cpu copies).
    std::vector<float> forward(const float * latent, int64_t T, float t,
                               const float * clip_f, int64_t clip_S,
                               const float * sync_f, int64_t sync_S,
                               const float * text_f,
                               const float * t5,
                               const float * global_text,
                               std::map<std::string, std::vector<float>> * dbg = nullptr);

    // Full rectified-flow Euler sampler with classifier-free guidance (text-only).
    // noise/return are [T, latent_dim] (ne[0]=T). Uncond uses learned empties.
    // If flow0 != null, fills it with the CFG velocity at step 0 ([T,latent_dim]).
    std::vector<float> sample(const float * noise, int64_t T,
                              const float * clip_f, int64_t clip_S,
                              const float * sync_f, int64_t sync_S,
                              const float * text_f,
                              const float * t5,
                              const float * global_text,
                              int steps, float cfg_scale,
                              std::vector<float> * flow0 = nullptr);
};
