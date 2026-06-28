#include "vae_decoder.h"
#include "common/ts_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cstdio>

bool ts_vae::load(const std::string & gguf_path) {
    backend = ts_backend_init();  // GPU: transpose conv uses ggml_conv_transpose_1d (CUDA kernel)
    return model.load_backend(gguf_path, backend);
}
ts_vae::~ts_vae() { if (backend) ggml_backend_free(backend); }

namespace {

ggml_tensor * snake(ggml_context * c, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) {
    // x:[T,C]; alpha,beta:[C] -> reshape [1,C]; snake = x + sin^2(exp(a)*x) * exp(-b)
    const int64_t C = x->ne[1];
    ggml_tensor * a = ggml_reshape_2d(c, alpha, 1, C);
    ggml_tensor * b = ggml_reshape_2d(c, beta,  1, C);
    ggml_tensor * ea = ggml_exp(c, a);
    ggml_tensor * eb = ggml_exp(c, ggml_neg(c, b));         // exp(-beta) = 1/exp(beta)
    ggml_tensor * s  = ggml_sin(c, ggml_mul(c, x, ea));
    ggml_tensor * s2 = ggml_sqr(c, s);
    return ggml_add(c, x, ggml_mul(c, s2, eb));
}

ggml_tensor * conv1d(ggml_context * c, ggml_tensor * w, ggml_tensor * b,
                     ggml_tensor * x, int stride, int pad, int dil) {
    // w:[K,IC,OC]  x:[T,IC] -> [OL,OC].  Explicit F32 im2col (ggml_conv_1d forces F16).
    ggml_tensor * im = ggml_im2col(c, w, x, stride, 0, pad, 0, dil, 0, false, GGML_TYPE_F32); // [IC*K, OL, 1]
    ggml_tensor * cols = ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]);            // [IC*K, OL]
    ggml_tensor * wr   = ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]);                // [IC*K, OC]
    ggml_tensor * y    = ggml_mul_mat(c, cols, wr);                                           // [OL, OC]
    if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0]));
    return y;
}

ggml_tensor * conv_t1d(ggml_context * c, ggml_tensor * w, ggml_tensor * b,
                       ggml_tensor * x, int stride) {
    // Transpose conv == zero-stuff upsample by `stride` + regular stride-1 conv1d with the
    // flipped/channel-swapped kernel (done at conversion). Runs as pad + im2col + GEMM on CUDA;
    // the native CUDA conv_transpose_1d kernel is O(out*IC*OC*T_in) -- unusable for x2048 upsample.
    // w: ggml ne=[K, IC, OC] (K=2*stride).  x:[T_in, IC] -> [T_in*stride, OC].
    const int     K    = (int) w->ne[0];               // 2*stride
    const int     pad  = K - 1 - stride / 2;            // conv1d pad' = K-1-p, with p=stride/2
    const int64_t T_in = x->ne[0];
    const int64_t IC   = x->ne[1];
    // Insert stride-1 zeros after each time sample. ggml_pad maps ne1->gridDim.y (max 65535), so
    // pad the *stride* axis (small) with time kept in ne0/gridDim.x, then permute+cont to the
    // interleaved [stride, T_in, IC] layout (cont uses a flat grid -> safe for long sequences).
    ggml_tensor * x3 = ggml_reshape_3d(c, x, T_in, 1, IC);           // [T_in, 1, IC]
    ggml_tensor * P  = ggml_pad(c, x3, 0, stride - 1, 0, 0);          // [T_in, stride, IC], P[:,0,:]=x
    ggml_tensor * Pp = ggml_cont(c, ggml_permute(c, P, 1, 0, 2, 3));  // [stride, T_in, IC]
    ggml_tensor * uf = ggml_reshape_2d(c, Pp, stride * T_in, IC);     // [stride*T_in, IC] zero-stuffed
    // drop trailing zeros so length is (T_in-1)*stride+1 -> conv1d yields exactly T_in*stride
    ggml_tensor * u  = ggml_cont(c, ggml_view_2d(c, uf, (T_in - 1) * stride + 1, IC, uf->nb[1], 0));
    return conv1d(c, w, b, u, 1, pad, 1);                            // [T_in*stride, OC]
}

} // namespace

int64_t ts_vae::decode(const float * latent, int64_t T, std::vector<float> & audio) {
    auto W = [&](const std::string & n) { return model.need(n); };

    // graph context
    const size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> buf(ctx_size);
    ggml_init_params ip{ ctx_size, buf.data(), true };
    ggml_context * c = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, 64);
    ggml_set_name(inp, "latent");
    ggml_set_input(inp);

    // conv1: [T,64] -> [T,2048]  (k7, pad3)
    ggml_tensor * x = conv1d(c, W("decoder.layers.0.weight"), W("decoder.layers.0.bias"), inp, 1, 3, 1);

    // 5 DecoderBlocks (layers.1..5): snake -> convT -> 3 resunits
    const int strides[5] = {8, 8, 4, 4, 2};
    const int dils[3]    = {1, 3, 9};
    for (int n = 0; n < 5; ++n) {
        const std::string bp = "decoder.layers." + std::to_string(n + 1);
        const int st  = strides[n];
        x = snake(c, x, W(bp + ".layers.0.alpha"), W(bp + ".layers.0.beta"));
        ggml_tensor * ctw = W(bp + ".layers.1.weight");          // ggml ne=[K, OC, IC]
        x = conv_t1d(c, ctw, W(bp + ".layers.1.bias"), x, st);
        for (int r = 0; r < 3; ++r) {
            const std::string rp = bp + ".layers." + std::to_string(r + 2);
            ggml_tensor * skip = x;
            ggml_tensor * h = snake(c, x, W(rp + ".layers.0.alpha"), W(rp + ".layers.0.beta"));
            h = conv1d(c, W(rp + ".layers.1.weight"), W(rp + ".layers.1.bias"), h, 1, 3 * dils[r], dils[r]);
            h = snake(c, h, W(rp + ".layers.2.alpha"), W(rp + ".layers.2.beta"));
            h = conv1d(c, W(rp + ".layers.3.weight"), W(rp + ".layers.3.bias"), h, 1, 0, 1);
            x = ggml_add(c, skip, h);
        }
    }

    // final: snake -> conv2 (k7, pad3, no bias) -> [T_audio, 2]
    x = snake(c, x, W("decoder.layers.6.alpha"), W("decoder.layers.6.beta"));
    x = conv1d(c, W("decoder.layers.7.weight"), nullptr, x, 1, 3, 1);
    ggml_set_name(x, "audio");
    ggml_set_output(x);

    // build + alloc + compute (GPU)
    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "ts_vae: gallocr alloc failed\n");
        ggml_gallocr_free(alloc); ggml_free(c); return -1;
    }
    ggml_backend_tensor_set(inp, latent, 0, (size_t) T * 64 * sizeof(float));
    fprintf(stderr, "ts_vae: graph %d nodes, computing...\n", ggml_graph_n_nodes(gf));
    ggml_backend_graph_compute(backend, gf);

    const int64_t T_audio = x->ne[0];
    audio.resize((size_t) T_audio * 2);
    ggml_backend_tensor_get(x, audio.data(), 0, audio.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(c);
    return T_audio;
}
