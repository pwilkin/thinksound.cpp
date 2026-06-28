// ts-encoders_test <t5.gguf> <metaclip.gguf> <golden_text.gguf>
#include "t5_encoder.h"
#include "clip_text.h"
#include "common/ts_model.h"
#include "ggml.h"
#include <cstdio>
#include <cmath>
#include <vector>

static void cmp(const char * name, const std::vector<float> & a, const ggml_tensor * g) {
    const float * b = (const float *) g->data;
    int64_t n = std::min<int64_t>((int64_t) a.size(), ggml_nelements(g));
    double mx = 0, ss = 0, rs = 0;
    for (int64_t i = 0; i < n; ++i) { double d=(double)a[i]-(double)b[i]; mx=std::max(mx,std::fabs(d)); ss+=d*d; rs+=(double)b[i]*b[i]; }
    printf("  %-22s n=%-7ld max|d|=%.4e rel_l2=%.4e %s\n", name,(long)n,mx,std::sqrt(ss/(rs+1e-30)),
           std::sqrt(ss/(rs+1e-30))<5e-3?"OK":"**");
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr,"usage: ts-encoders_test <t5.gguf> <metaclip.gguf> <golden_text.gguf>\n"); return 1; }
    ts_model g; if (!g.load(argv[3])) return 1;
    ggml_tensor * t5ids = g.need("t5_input_ids");      // [77]
    ggml_tensor * clids = g.need("clip_input_ids");    // [77]
    int nt = (int) t5ids->ne[0], nc = (int) clids->ne[0];

    ts_t5 t5; if (!t5.load(argv[1])) return 1;
    std::vector<float> t5out;
    t5.encode((const int32_t*) t5ids->data, nt, t5out);

    ts_clip_text clip; if (!clip.load(argv[2])) return 1;
    std::vector<float> per_token, global;
    clip.encode((const int32_t*) clids->data, nc, per_token, global);

    printf("=== encoder parity vs golden_text ===\n");
    cmp("t5_features", t5out, g.need("t5_features"));
    cmp("metaclip_text_features", per_token, g.need("metaclip_text_features"));
    cmp("metaclip_global", global, g.need("metaclip_global_text_features"));
    return 0;
}
