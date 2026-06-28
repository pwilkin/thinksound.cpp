#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>

// Initialize the compute backend. want_gpu=true prefers the GPU device with the most
// memory (f32 weights land on the 16GB card); want_gpu=false forces CPU. Override with
// TS_BACKEND=CUDA0/CPU/... TS_FORCE_CPU=1 forces CPU for every model.
inline ggml_backend_t ts_backend_init(bool want_gpu = true) {
    if (getenv("TS_FORCE_CPU")) want_gpu = false;
    if (!want_gpu) {
        ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        fprintf(stderr, "[backend] %s (cpu)\n", ggml_backend_name(b));
        return b;
    }
    if (const char * force = getenv("TS_BACKEND")) {
        if (ggml_backend_t b = ggml_backend_init_by_name(force, nullptr)) {
            fprintf(stderr, "[backend] %s (forced)\n", ggml_backend_name(b));
            return b;
        }
        fprintf(stderr, "[backend] TS_BACKEND=%s not found, falling back\n", force);
    }
    ggml_backend_dev_t best = nullptr; size_t best_mem = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t fr = 0, tot = 0; ggml_backend_dev_memory(d, &fr, &tot);
        enum ggml_backend_dev_type ty = ggml_backend_dev_type(d);
        if (ty != GGML_BACKEND_DEVICE_TYPE_GPU && ty != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (tot > best_mem) { best_mem = tot; best = d; }
    }
    ggml_backend_t b = best ? ggml_backend_dev_init(best, nullptr)
                            : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    fprintf(stderr, "[backend] %s (%zu MB total)\n", ggml_backend_name(b), best_mem / (1024 * 1024));
    return b;
}
