#include "common/ts_model.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

ts_model::~ts_model() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (gguf) gguf_free(gguf);
    if (ctx)  ggml_free(ctx);
}

bool ts_model::load_backend(const std::string & path, ggml_backend_t backend) {
    // metadata-only load: tensors created in ctx without data
    gguf_init_params p{}; p.no_alloc = true; p.ctx = &ctx;
    gguf = gguf_init_from_file(path.c_str(), p);
    if (!gguf) { fprintf(stderr, "ts_model: failed to load gguf '%s'\n", path.c_str()); return false; }
    // allocate all weight tensors in a backend buffer (GPU or CPU)
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { fprintf(stderr, "ts_model: backend tensor alloc failed for '%s'\n", path.c_str()); return false; }
    // stream tensor data from file into the backend buffer
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "ts_model: cannot reopen '%s'\n", path.c_str()); return false; }
    const size_t data_off = gguf_get_data_offset(gguf);
    std::vector<uint8_t> host;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        const char * name = ggml_get_name(t);
        int64_t idx = gguf_find_tensor(gguf, name);
        if (idx < 0) continue;
        size_t off = gguf_get_tensor_offset(gguf, idx);
        size_t sz  = ggml_nbytes(t);
        host.resize(sz);
        if (fseek(f, (long) (data_off + off), SEEK_SET) != 0 || fread(host.data(), 1, sz, f) != sz) {
            fprintf(stderr, "ts_model: read failed for tensor '%s'\n", name); fclose(f); return false;
        }
        ggml_backend_tensor_set(t, host.data(), 0, sz);
        tensors[name] = t;
    }
    fclose(f);
    return true;
}

void ts_model::get_data(ggml_tensor * t, void * dst) const {
    ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t));
}

bool ts_model::load(const std::string & path) {
    gguf_init_params p{};
    p.no_alloc = false;     // allocate + load tensor data into ctx (CPU)
    p.ctx      = &ctx;
    gguf = gguf_init_from_file(path.c_str(), p);
    if (!gguf) {
        fprintf(stderr, "ts_model: failed to load gguf '%s'\n", path.c_str());
        return false;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        tensors[ggml_get_name(t)] = t;
    }
    return true;
}

ggml_tensor * ts_model::get(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * ts_model::need(const std::string & name) const {
    ggml_tensor * t = get(name);
    if (!t) {
        fprintf(stderr, "ts_model: required tensor missing: '%s'\n", name.c_str());
        abort();
    }
    return t;
}

bool ts_model::has_kv(const std::string & key) const {
    return gguf_find_key(gguf, key.c_str()) >= 0;
}

int32_t ts_model::get_i32(const std::string & key, int32_t def) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(gguf, id);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(gguf, id) ? 1 : 0;
        default: return def;
    }
}

float ts_model::get_f32(const std::string & key, float def) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(gguf, id);
        case GGUF_TYPE_INT32:   return (float) gguf_get_val_i32(gguf, id);
        default: return def;
    }
}

std::string ts_model::get_str(const std::string & key, const std::string & def) const {
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf, id);
}

std::vector<int32_t> ts_model::get_arr_i32(const std::string & key) const {
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    size_t n = gguf_get_arr_n(gguf, id);
    enum gguf_type at = gguf_get_arr_type(gguf, id);
    const void * data = gguf_get_arr_data(gguf, id);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (at == GGUF_TYPE_INT32)       out[i] = ((const int32_t *) data)[i];
        else if (at == GGUF_TYPE_UINT32) out[i] = (int32_t) ((const uint32_t *) data)[i];
        else if (at == GGUF_TYPE_INT64)  out[i] = (int32_t) ((const int64_t *) data)[i];
        else                             out[i] = 0;
    }
    return out;
}

std::vector<float> ts_model::get_arr_f32(const std::string & key) const {
    std::vector<float> out;
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    size_t n = gguf_get_arr_n(gguf, id);
    const void * data = gguf_get_arr_data(gguf, id);
    enum gguf_type at = gguf_get_arr_type(gguf, id);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (at == GGUF_TYPE_FLOAT32)      out[i] = ((const float *) data)[i];
        else if (at == GGUF_TYPE_FLOAT64) out[i] = (float) ((const double *) data)[i];
    }
    return out;
}

std::vector<std::string> ts_model::get_arr_str(const std::string & key) const {
    std::vector<std::string> out;
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gguf, id) != GGUF_TYPE_STRING) return out;
    size_t n = gguf_get_arr_n(gguf, id);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(gguf, id, i));
    return out;
}

std::string ts_shape_str(const ggml_tensor * t) {
    char buf[64];
    snprintf(buf, sizeof buf, "%ld,%ld,%ld,%ld",
             (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], (long) t->ne[3]);
    return buf;
}
