// ts-dit_test <dit.gguf> <golden_dit.gguf> <golden_dit_dbg.gguf>
//   single cond-only forward (t=1, cfg=1), compare each stage to the debug golden.
#include "mmdit.h"
#include "common/ts_model.h"
#include "ggml.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

static void cmp(const char * name, const std::vector<float> & a, const ggml_tensor * g) {
    const float * b = (const float *) g->data;
    int64_t n = std::min<int64_t>((int64_t) a.size(), ggml_nelements(g));
    double mx = 0, ss = 0, rs = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = (double) a[i] - (double) b[i];
        mx = std::max(mx, std::fabs(d)); ss += d * d; rs += (double) b[i] * b[i];
    }
    printf("  %-14s n=%-7ld  max|d|=%.5e  rel_l2=%.5e  %s\n", name, (long) n,
           mx, std::sqrt(ss / (rs + 1e-30)), (std::sqrt(ss/(rs+1e-30)) < 2e-3 ? "OK" : "**"));
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: ts-dit_test <dit.gguf> <golden_dit.gguf> <golden_dit_dbg.gguf>\n"); return 1; }
    ts_dit dit; if (!dit.load(argv[1])) return 1;
    ts_model gin, gdb;
    if (!gin.load(argv[2]) || !gdb.load(argv[3])) return 1;

    ggml_tensor * noise = gin.need("noise");          // [T,64]
    int64_t T = noise->ne[0];
    ggml_tensor * clip = gin.need("clip_f");          // [1024,72]
    ggml_tensor * sync = gin.need("sync_f");          // [768,216]
    ggml_tensor * text = gin.need("text_f");          // [1024,77]
    ggml_tensor * t5   = gin.need("t5_features");     // [2048,77]
    ggml_tensor * gt   = gin.need("metaclip_global_text_features"); // [1024]

    std::map<std::string, std::vector<float>> dbg;
    std::vector<float> flow = dit.forward(
        (const float*) noise->data, T, 1.0f,
        (const float*) clip->data, clip->ne[1],
        (const float*) sync->data, sync->ne[1],
        (const float*) text->data,
        (const float*) t5->data,
        (const float*) gt->data, &dbg);

    // per-stream breakdown of a0_v [d+64h+1024n]: latent n[0:194] clip[194:266] text[266:420]
    if (gdb.get("a0_v")) {
        const auto & a = dbg["a0_v"]; const float * b = (const float*) gdb.need("a0_v")->data;
        struct { const char* nm; int lo, hi; } st[] = {{"  a0_v.latent",0,194},{"  a0_v.clip",194,266},{"  a0_v.text",266,420}};
        for (auto & s : st) { double ss=0,rs=0; for (int n=s.lo;n<s.hi;++n) for (int i=0;i<1024;++i){ int idx=i+1024*n; double d=a[idx]-b[idx]; ss+=d*d; rs+=(double)b[idx]*b[idx]; } printf("%s rel_l2=%.4e\n", s.nm, std::sqrt(ss/(rs+1e-30))); }
    }
    printf("=== stage parity vs debug golden ===\n");
    cmp("x_in",        dbg["x_in"],        gdb.need("x_in"));
    cmp("t_emb",       dbg["t_emb"],       gdb.need("t_emb"));
    for (const char * nm : {"clip_pp","text_pp","clip_fc","text_fc","gcmlp","global_c","extended_c","text_norm1","text_xn","a0_q","a0_k","a0_v","a0_out","jb0_lat","jb0_clip","jb0_text"})
        if (gdb.get(nm)) cmp(nm, dbg[nm], gdb.need(nm));
    cmp("after_joint", dbg["after_joint"], gdb.need("after_joint"));
    cmp("after_fused", dbg["after_fused"], gdb.need("after_fused"));
    cmp("flow",        flow,               gdb.need("flow_cl"));
    return 0;
}
