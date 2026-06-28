// ts-generate — standalone text -> sound-effect.
//   tokenize (CLIP-BPE caption + T5 CoT) -> encoders -> conditioning (video=empties)
//   -> MM-DiT rectified-flow sampler -> oobleck VAE decode -> wav.
//   Pass --ref <golden_e2e.gguf> to reuse the reference noise and print parity.
#include "mmdit.h"
#include "vae_decoder.h"
#include "t5_encoder.h"
#include "clip_text.h"
#include "tokenizer.h"
#include "common/ts_model.h"
#include "common/ts_wav.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

static const char * opt(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}
static void cmp(const char * name, const std::vector<float> & a, const ggml_tensor * g) {
    const float * b = (const float *) g->data; int64_t n = std::min<int64_t>((int64_t) a.size(), ggml_nelements(g));
    double mx = 0, ss = 0, rs = 0;
    for (int64_t i = 0; i < n; ++i) { double d = (double)a[i]-(double)b[i]; mx=std::max(mx,std::fabs(d)); ss+=d*d; rs+=(double)b[i]*b[i]; }
    printf("  %-7s rel_l2=%.4e max|d|=%.4e %s\n", name, std::sqrt(ss/(rs+1e-30)), mx, std::sqrt(ss/(rs+1e-30))<1e-2?"OK":"**");
}

int main(int argc, char ** argv) {
    std::string g = opt(argc, argv, "--dir", "/media/ilintar/D_SSD/thinksound/gguf");
    std::string dit_s=g+"/dit-f32.gguf", vae_s=g+"/vae-f32.gguf", t5_s=g+"/t5-f32.gguf",
                clip_s=g+"/metaclip-text-f32.gguf", t5t_s=g+"/t5-tokenizer.gguf", clt_s=g+"/clip-tokenizer.gguf";
    const char * dit_p  = opt(argc, argv, "--dit",  dit_s.c_str());
    const char * vae_p  = opt(argc, argv, "--vae",  vae_s.c_str());
    const char * t5_p   = opt(argc, argv, "--t5",   t5_s.c_str());
    const char * clip_p = opt(argc, argv, "--clip", clip_s.c_str());
    const char * t5tok_p   = opt(argc, argv, "--t5-tok",   t5t_s.c_str());
    const char * cliptok_p = opt(argc, argv, "--clip-tok", clt_s.c_str());
    const char * caption = opt(argc, argv, "--caption", "a dog barking");
    const char * cot     = opt(argc, argv, "--cot", "A dog barks several times, sharp and clear, in a quiet room.");
    const char * out_wav = opt(argc, argv, "-o", "out.wav");
    const char * ref_p   = opt(argc, argv, "--ref", nullptr);
    int   steps = atoi(opt(argc, argv, "--steps", "24"));
    float cfg   = atof(opt(argc, argv, "--cfg", "5"));
    unsigned seed = (unsigned) atoi(opt(argc, argv, "--seed", "4242"));

    // Duration -> sequence lengths (reference unwrap.py): the conditioning streams are
    // integer-second framecounts (clip 8 fps, sync 24 fps; sync_S must stay /8-divisible),
    // the latent runs at 44100/2048 Hz. With --ref the golden's noise pins T (ignore --duration).
    int dur = (int) lround(atof(opt(argc, argv, "--duration", "9")));
    if (dur < 1) dur = 1;  if (dur > 30) dur = 30;   // clamp: >30 s risks VAE OOM on 16 GB
    int64_t T = (int64_t) llround(44100.0 / 2048.0 * dur), clip_S = 8 * dur, sync_S = 24 * dur;
    if (ref_p) { T = 194; clip_S = 8 * 9; sync_S = 24 * 9; dur = 9; }  // parity uses the 9 s golden
    printf("duration=%d s  (T=%lld, clip_S=%lld, sync_S=%lld)\n", dur, (long long)T, (long long)clip_S, (long long)sync_S);

    ts_clip_tokenizer clip_tok; ts_t5_tokenizer t5_tok;
    if (!clip_tok.load(cliptok_p) || !t5_tok.load(t5tok_p)) return 1;
    std::vector<int32_t> clip_ids = clip_tok.encode(caption, 77);
    std::vector<int32_t> t5_ids   = t5_tok.encode(cot, 77);
    printf("caption=\"%s\"  cot=\"%s\"\n", caption, cot);

    // noise: from reference golden if --ref (for parity), else seeded randn (host, kept for parity)
    ts_model ref; std::vector<float> noise_buf;
    const float * noise;
    if (ref_p && ref.load(ref_p)) { noise = (const float *) ref.need("noise")->data; }
    else { std::mt19937 rng(seed); std::normal_distribution<float> nd(0,1); noise_buf.resize((size_t)T*64);
           for (auto & x : noise_buf) x = nd(rng); noise = noise_buf.data(); }

    // Each network is loaded, used, then freed before the next so peak VRAM is one stage's
    // weights + intermediates, not the sum (all four would OOM a 16 GB card at VAE decode).
    printf("encoding...\n");
    std::vector<float> t5feat, mt, mg;
    { ts_t5 t5; if (!t5.load(t5_p)) return 1; t5.encode(t5_ids.data(), 77, t5feat); }       // [2048,77]
    { ts_clip_text clipm; if (!clipm.load(clip_p)) return 1; clipm.encode(clip_ids.data(), 77, mt, mg); } // [1024,77],[1024]

    printf("sampling (%d steps, cfg %.1f)...\n", steps, cfg);
    std::vector<float> latent;
    {
        ts_dit dit; if (!dit.load(dit_p)) return 1;
        // conditioning: clip/sync = learned empties (text-only)
        ggml_tensor * ec = dit.model.need("empty_clip_feat"); ggml_tensor * es = dit.model.need("empty_sync_feat");
        std::vector<float> ecv(1024), esv(768); dit.model.get_data(ec, ecv.data()); dit.model.get_data(es, esv.data());
        std::vector<float> clip_f((size_t)1024*clip_S), sync_f((size_t)768*sync_S);
        for (int s = 0; s < clip_S; ++s) for (int i = 0; i < 1024; ++i) clip_f[(size_t)s*1024+i] = ecv[i];
        for (int s = 0; s < sync_S; ++s) for (int i = 0; i < 768;  ++i) sync_f[(size_t)s*768 +i] = esv[i];
        latent = dit.sample(noise, T, clip_f.data(), clip_S, sync_f.data(), sync_S,
                            mt.data(), t5feat.data(), mg.data(), steps, cfg);
    }

    std::vector<float> audio; int64_t Ta;
    { ts_vae vae; if (!vae.load(vae_p)) return 1; Ta = vae.decode(latent.data(), T, audio); }

    if (ref_p && ref.gguf) { printf("=== parity vs reference ===\n");
        cmp("latent", latent, ref.need("latent"));
        if (Ta > 0 && ref.get("wav_ref")) cmp("wav", audio, ref.need("wav_ref"));
        else if (Ta <= 0) printf("  wav     SKIPPED (decode failed)\n"); }
    if (Ta > 0) { ts_write_wav(out_wav, audio.data(), 2, Ta, 44100, true); printf("wrote %s (%.2fs @ 44.1kHz)\n", out_wav, Ta/44100.0); }
    return 0;
}
