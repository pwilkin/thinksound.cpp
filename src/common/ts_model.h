#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

// GGUF model holder. load() keeps tensor data on the host (CPU ctx) — used for
// golden/test files. load_backend() places weights in a backend buffer (CUDA/CPU)
// for GPU compute. tensor ->data is host-readable only after load(); use
// get_data() to copy out of a backend tensor.
struct ts_model {
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr;
    ggml_backend_buffer_t buffer = nullptr;   // weight buffer when load_backend used
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~ts_model();
    bool load(const std::string & path);                       // host (CPU) data
    bool load_backend(const std::string & path, ggml_backend_t backend); // weights on backend

    void get_data(ggml_tensor * t, void * dst) const;          // copy tensor -> host (CPU or GPU)

    ggml_tensor * get(const std::string & name) const;   // nullptr if missing
    ggml_tensor * need(const std::string & name) const;  // abort if missing

    bool                 has_kv(const std::string & key) const;
    int32_t              get_i32(const std::string & key, int32_t def = 0) const;
    float                get_f32(const std::string & key, float def = 0.f) const;
    std::string          get_str(const std::string & key, const std::string & def = "") const;
    std::vector<int32_t>     get_arr_i32(const std::string & key) const;
    std::vector<float>       get_arr_f32(const std::string & key) const;
    std::vector<std::string> get_arr_str(const std::string & key) const;
};

// shape as "a,b,c" in ggml (ne) order
std::string ts_shape_str(const ggml_tensor * t);
