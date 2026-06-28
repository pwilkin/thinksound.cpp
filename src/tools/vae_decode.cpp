// ts-vae_decode <vae.gguf> <golden.gguf> [out.wav]
//   decode golden `latent` with the ggml VAE, compare to golden `wav_ref`, write wav.
#include "vae_decoder.h"
#include "common/ts_wav.h"
#include "ggml.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: ts-vae_decode <vae.gguf> <golden.gguf> [out.wav]\n"); return 1; }
    const char * out_wav = argc > 3 ? argv[3] : "ts_vae_out.wav";

    ts_vae vae;
    if (!vae.load(argv[1])) return 1;

    ts_model golden;
    if (!golden.load(argv[2])) return 1;
    ggml_tensor * lat = golden.need("latent");     // ne=[T,64]
    ggml_tensor * ref = golden.get("wav_ref");      // ne=[T_audio,2] (optional)
    const int64_t T = lat->ne[0];
    printf("latent T=%ld C=%ld\n", (long) T, (long) lat->ne[1]);

    std::vector<float> audio;
    int64_t T_audio = vae.decode((const float *) lat->data, T, audio);
    if (T_audio < 0) return 1;
    printf("decoded T_audio=%ld (%.2fs @ 44.1kHz)\n", (long) T_audio, T_audio / 44100.0);

    if (ref) {
        const int64_t Tr = ref->ne[0];
        const float * r = (const float *) ref->data;
        int64_t n = std::min<int64_t>(T_audio, Tr) * 2;
        double maxabs = 0, sumsq = 0, refsq = 0;
        for (int64_t i = 0; i < n; ++i) {
            double d = (double) audio[i] - (double) r[i];
            maxabs = std::max(maxabs, std::fabs(d));
            sumsq += d * d; refsq += (double) r[i] * r[i];
        }
        double rmse = std::sqrt(sumsq / n);
        double rel  = std::sqrt(sumsq / (refsq + 1e-20));
        printf("PARITY vs wav_ref: max|d|=%.6f  rmse=%.6f  rel_l2=%.6f  (n=%ld, T_ref=%ld)\n",
               maxabs, rmse, rel, (long) n, (long) Tr);
    }

    ts_write_wav(out_wav, audio.data(), 2, T_audio, 44100, true);
    printf("wrote %s\n", out_wav);
    return 0;
}
