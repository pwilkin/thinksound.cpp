#pragma once
#include "common/ts_model.h"
#include <vector>
#include <string>

// Oobleck (stable-audio) VAE decoder on ggml (CPU backend).
// Input:  latent [T, 64] in ggml time-major layout (data[c*T + t]).
// Output: audio  [2, T_audio] planar (ch-major: data[c*T_audio + t]), T_audio = T*2048.
struct ts_vae {
    ts_model model;
    ggml_backend_t backend = nullptr;
    bool load(const std::string & gguf_path);
    ~ts_vae();

    // Decode; fills audio (resized to 2*T_audio) and returns T_audio (or -1).
    int64_t decode(const float * latent, int64_t T, std::vector<float> & audio);
};
