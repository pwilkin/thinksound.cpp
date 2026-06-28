// ts-server — resident HTTP wrapper around the ThinkSound text->sound-effect pipeline.
//
//   GET  /health           -> "ok"
//   POST /generate         JSON in, audio/wav out
//     body: { "caption": str (required), "cot"|"description": str,
//             "duration": int (1..30), "steps": int, "cfg": float, "seed": int }
//
// The model GGUFs are resolved once at startup and kept warm; each request runs
// the same sequential load/free pipeline as ts-generate (peak VRAM = one stage),
// serialized by a mutex. The compute backend is chosen by ts_backend_init via the
// usual TS_BACKEND / CUDA_VISIBLE_DEVICES environment (Vulkan when CUDA is hidden).
#include "mmdit.h"
#include "vae_decoder.h"
#include "t5_encoder.h"
#include "clip_text.h"
#include "tokenizer.h"
#include "common/ts_model.h"
#include "common/ts_wav.h"
#include "ggml.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct Paths {
    std::string dit, vae, t5, clip, t5tok, cliptok;
};

const char * opt(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

// Run the full text->wav pipeline. Returns the encoded 16-bit WAV bytes, or an
// empty vector on failure (with *err set). Mirrors ts-generate's stage order.
std::vector<uint8_t> generate_wav(const Paths & p,
                                  const ts_clip_tokenizer & clip_tok,
                                  const ts_t5_tokenizer & t5_tok,
                                  const std::string & caption, const std::string & cot,
                                  int dur, int steps, float cfg, unsigned seed,
                                  std::string * err) {
    if (dur < 1) dur = 1;
    if (dur > 30) dur = 30;
    int64_t T = (int64_t) llround(44100.0 / 2048.0 * dur), clip_S = 8 * dur, sync_S = 24 * dur;

    std::vector<int32_t> clip_ids = clip_tok.encode(caption.c_str(), 77);
    std::vector<int32_t> t5_ids   = t5_tok.encode(cot.c_str(), 77);

    std::vector<float> noise_buf((size_t) T * 64);
    { std::mt19937 rng(seed); std::normal_distribution<float> nd(0, 1);
      for (auto & x : noise_buf) x = nd(rng); }
    const float * noise = noise_buf.data();

    std::vector<float> t5feat, mt, mg;
    { ts_t5 t5; if (!t5.load(p.t5.c_str())) { *err = "t5 load failed"; return {}; }
      t5.encode(t5_ids.data(), 77, t5feat); }
    { ts_clip_text clipm; if (!clipm.load(p.clip.c_str())) { *err = "clip load failed"; return {}; }
      clipm.encode(clip_ids.data(), 77, mt, mg); }

    std::vector<float> latent;
    {
        ts_dit dit; if (!dit.load(p.dit.c_str())) { *err = "dit load failed"; return {}; }
        ggml_tensor * ec = dit.model.need("empty_clip_feat");
        ggml_tensor * es = dit.model.need("empty_sync_feat");
        std::vector<float> ecv(1024), esv(768);
        dit.model.get_data(ec, ecv.data());
        dit.model.get_data(es, esv.data());
        std::vector<float> clip_f((size_t) 1024 * clip_S), sync_f((size_t) 768 * sync_S);
        for (int s = 0; s < clip_S; ++s) for (int i = 0; i < 1024; ++i) clip_f[(size_t) s * 1024 + i] = ecv[i];
        for (int s = 0; s < sync_S; ++s) for (int i = 0; i < 768;  ++i) sync_f[(size_t) s * 768  + i] = esv[i];
        latent = dit.sample(noise, T, clip_f.data(), clip_S, sync_f.data(), sync_S,
                            mt.data(), t5feat.data(), mg.data(), steps, cfg);
    }

    std::vector<float> audio; int64_t Ta = 0;
    { ts_vae vae; if (!vae.load(p.vae.c_str())) { *err = "vae load failed"; return {}; }
      Ta = vae.decode(latent.data(), T, audio); }
    if (Ta <= 0) { *err = "vae decode failed"; return {}; }

    // ts_write_wav only writes a file; render to a temp file then read the bytes.
    std::string tmp = std::string(std::tmpnam(nullptr)) + ".wav";
    if (!ts_write_wav(tmp, audio.data(), 2, Ta, 44100, true)) { *err = "wav encode failed"; return {}; }
    std::ifstream f(tmp, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    std::remove(tmp.c_str());
    return bytes;
}

} // namespace

int main(int argc, char ** argv) {
    std::string g = opt(argc, argv, "--dir", "/media/ilintar/D_SSD/thinksound/gguf");
    Paths p{
        opt(argc, argv, "--dit",      (g + "/dit-bf16.gguf").c_str()),
        opt(argc, argv, "--vae",      (g + "/vae-f32.gguf").c_str()),
        opt(argc, argv, "--t5",       (g + "/t5-bf16.gguf").c_str()),
        opt(argc, argv, "--clip",     (g + "/metaclip-text-f32.gguf").c_str()),
        opt(argc, argv, "--t5-tok",   (g + "/t5-tokenizer.gguf").c_str()),
        opt(argc, argv, "--clip-tok", (g + "/clip-tokenizer.gguf").c_str()),
    };
    std::string host = opt(argc, argv, "--host", "127.0.0.1");
    int port = atoi(opt(argc, argv, "--port", "8080"));

    ts_clip_tokenizer clip_tok; ts_t5_tokenizer t5_tok;
    if (!clip_tok.load(p.cliptok.c_str()) || !t5_tok.load(p.t5tok.c_str())) {
        std::fprintf(stderr, "[ts-server] failed to load tokenizers\n");
        return 1;
    }
    std::fprintf(stderr, "[ts-server] tokenizers loaded; models dir=%s\n", g.c_str());

    std::mutex gen_mu;
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("ok", "text/plain");
    });

    svr.Post("/generate", [&](const httplib::Request & rq, httplib::Response & res) {
        json body;
        try { body = json::parse(rq.body); }
        catch (const std::exception & e) {
            res.status = 400;
            res.set_content(std::string("{\"error\":\"invalid JSON: ") + e.what() + "\"}", "application/json");
            return;
        }
        auto gets = [&](const char * k, const std::string & d) {
            return body.contains(k) && body[k].is_string() ? body[k].get<std::string>() : d;
        };
        std::string caption = gets("caption", body.contains("prompt") ? gets("prompt", "") : "");
        if (caption.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'caption'\"}", "application/json");
            return;
        }
        std::string cot = body.contains("cot") ? gets("cot", caption)
                        : body.contains("description") ? gets("description", caption) : caption;
        int   dur   = body.value("duration", 9);
        int   steps = body.value("steps", 24);
        float cfg   = body.value("cfg", 5.0f);
        unsigned seed = (unsigned) body.value("seed", 4242);

        std::string err;
        std::vector<uint8_t> wav;
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            std::fprintf(stderr, "[ts-server] generate caption=%zu chars dur=%d steps=%d cfg=%.1f\n",
                         caption.size(), dur, steps, cfg);
            wav = generate_wav(p, clip_tok, t5_tok, caption, cot, dur, steps, cfg, seed, &err);
        }
        if (wav.empty()) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + err + "\"}", "application/json");
            return;
        }
        res.set_content(reinterpret_cast<const char *>(wav.data()), wav.size(), "audio/wav");
    });

    std::fprintf(stderr, "[ts-server] listening on http://%s:%d\n", host.c_str(), port);
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "[ts-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
