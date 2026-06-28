// ts-gguf-dump <file.gguf>  — print metadata + tensor inventory. Validates the
// GGUF loader and the ggml link.
#include "common/ts_model.h"
#include <cstdio>
#include <string>

static const char * kv_to_str(gguf_context * g, int64_t id, std::string & scratch) {
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_INT32:  scratch = std::to_string(gguf_get_val_i32(g, id)); break;
        case GGUF_TYPE_UINT32: scratch = std::to_string(gguf_get_val_u32(g, id)); break;
        case GGUF_TYPE_INT64:  scratch = std::to_string(gguf_get_val_i64(g, id)); break;
        case GGUF_TYPE_FLOAT32:scratch = std::to_string(gguf_get_val_f32(g, id)); break;
        case GGUF_TYPE_FLOAT64:scratch = std::to_string(gguf_get_val_f64(g, id)); break;
        case GGUF_TYPE_BOOL:   scratch = gguf_get_val_bool(g, id) ? "true" : "false"; break;
        case GGUF_TYPE_STRING: scratch = gguf_get_val_str(g, id); break;
        case GGUF_TYPE_ARRAY:  scratch = "[array n=" + std::to_string(gguf_get_arr_n(g, id)) + "]"; break;
        default:               scratch = "?"; break;
    }
    return scratch.c_str();
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ts-gguf-dump <file.gguf>\n"); return 1; }
    ts_model m;
    if (!m.load(argv[1])) return 1;

    printf("gguf version %u, alignment %zu\n", gguf_get_version(m.gguf), gguf_get_alignment(m.gguf));
    int64_t nkv = gguf_get_n_kv(m.gguf);
    printf("=== %ld KV ===\n", (long) nkv);
    std::string scratch;
    for (int64_t i = 0; i < nkv; ++i) {
        printf("  %-40s = %s\n", gguf_get_key(m.gguf, i), kv_to_str(m.gguf, i, scratch));
    }

    int64_t nt = gguf_get_n_tensors(m.gguf);
    printf("=== %ld tensors ===\n", (long) nt);
    size_t total = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(m.gguf, i);
        ggml_tensor * t = m.get(name);
        size_t sz = gguf_get_tensor_size(m.gguf, i);
        total += sz;
        printf("  %-52s %-8s [%s]  %.2f MB\n", name, ggml_type_name(t->type),
               ts_shape_str(t).c_str(), sz / 1e6);
    }
    printf("total tensor bytes: %.2f MB\n", total / 1e6);
    return 0;
}
