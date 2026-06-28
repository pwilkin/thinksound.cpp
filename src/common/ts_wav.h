#pragma once
#include <string>
#include <cstdint>

// Write planar float audio (channel-major: ch0[0..n-1], ch1[0..n-1], ...) to a
// 16-bit PCM WAV. Normalization matches the ThinkSound reference:
//   x = x / max(|x|); clamp(-1,1); *32767  (global peak over all samples)
// If normalize == false, samples are assumed already in [-1,1].
bool ts_write_wav(const std::string & path,
                  const float * planar, int n_channels, int64_t n_frames,
                  int sample_rate, bool normalize = true);
