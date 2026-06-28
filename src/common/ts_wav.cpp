#include "common/ts_wav.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace {
void w32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }
void w16(FILE * f, uint16_t v) { fwrite(&v, 2, 1, f); }
}

bool ts_write_wav(const std::string & path,
                  const float * planar, int n_channels, int64_t n_frames,
                  int sample_rate, bool normalize) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ts_write_wav: cannot open %s\n", path.c_str()); return false; }

    float scale = 1.0f;
    if (normalize) {
        float peak = 0.f;
        for (int64_t i = 0; i < n_frames * n_channels; ++i) peak = std::max(peak, std::fabs(planar[i]));
        scale = peak > 0.f ? 1.0f / peak : 1.0f;
    }

    const uint32_t data_bytes = (uint32_t)(n_frames * n_channels * 2);
    const uint32_t byte_rate  = (uint32_t)(sample_rate * n_channels * 2);

    fwrite("RIFF", 1, 4, f); w32(f, 36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(f, 16); w16(f, 1); w16(f, (uint16_t) n_channels);
    w32(f, sample_rate); w32(f, byte_rate); w16(f, (uint16_t)(n_channels * 2)); w16(f, 16);
    fwrite("data", 1, 4, f); w32(f, data_bytes);

    // interleave channels: frame-major
    for (int64_t i = 0; i < n_frames; ++i) {
        for (int c = 0; c < n_channels; ++c) {
            float v = planar[(int64_t) c * n_frames + i] * scale;
            v = std::max(-1.0f, std::min(1.0f, v));
            int16_t s = (int16_t) std::lround(v * 32767.0f);
            w16(f, (uint16_t) s);
        }
    }
    fclose(f);
    return true;
}
