// ts-dit_sample <dit.gguf> <vae.gguf> <golden_dit.gguf> [out.wav]
//   full CFG rectified-flow sampler -> verify flow0 + latent -> VAE decode -> wav (M2+M3).
#include "mmdit.h"
#include "vae_decoder.h"
#include "common/ts_model.h"
#include "common/ts_wav.h"
#include "ggml.h"
#include <cstdio>
#include <cmath>
#include <vector>

static void cmp(const char * name, const std::vector<float> & a, const ggml_tensor * g) {
    const float * b = (const float *) g->data;
    int64_t n = std::min<int64_t>((int64_t) a.size(), ggml_nelements(g));
    double mx = 0, ss = 0, rs = 0;
    for (int64_t i = 0; i < n; ++i) { double d = (double)a[i]-(double)b[i]; mx=std::max(mx,std::fabs(d)); ss+=d*d; rs+=(double)b[i]*b[i]; }
    printf("  %-10s n=%-7ld max|d|=%.5e rel_l2=%.5e %s\n", name,(long)n,mx,std::sqrt(ss/(rs+1e-30)),
           std::sqrt(ss/(rs+1e-30))<5e-3?"OK":"**");
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr,"usage: ts-dit_sample <dit.gguf> <vae.gguf> <golden_dit.gguf> [out.wav]\n"); return 1; }
    const char * out_wav = argc>4?argv[4]:"ts_dit_out.wav";
    ts_dit dit; if (!dit.load(argv[1])) return 1;
    ts_vae vae; if (!vae.load(argv[2])) return 1;
    ts_model g; if (!g.load(argv[3])) return 1;

    ggml_tensor * noise = g.need("noise"); int64_t T = noise->ne[0];
    ggml_tensor * clip=g.need("clip_f"), *sync=g.need("sync_f"), *text=g.need("text_f"),
                * t5=g.need("t5_features"), *gt=g.need("metaclip_global_text_features");
    int steps = g.get_i32("gen.steps", 24);
    float cfg = g.get_f32("gen.cfg_scale", 5.0f);
    printf("sampling: steps=%d cfg=%.1f T=%ld\n", steps, cfg, (long)T);

    std::vector<float> flow0;
    std::vector<float> latent = dit.sample(
        (const float*)noise->data, T,
        (const float*)clip->data, clip->ne[1], (const float*)sync->data, sync->ne[1],
        (const float*)text->data, (const float*)t5->data, (const float*)gt->data,
        steps, cfg, &flow0);

    printf("=== parity vs golden ===\n");
    cmp("flow0", flow0, g.need("flow0"));
    cmp("latent", latent, g.need("latent"));

    std::vector<float> audio;
    int64_t Ta = vae.decode(latent.data(), T, audio);
    if (Ta > 0 && g.get("wav_ref")) cmp("wav", audio, g.need("wav_ref"));
    if (Ta > 0) { ts_write_wav(out_wav, audio.data(), 2, Ta, 44100, true); printf("wrote %s (%.2fs)\n", out_wav, Ta/44100.0); }
    return 0;
}
