#include "t5_encoder.h"
#include "common/ts_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cmath>

ts_t5::~ts_t5() { if (backend) ggml_backend_free(backend); }

namespace {
ggml_tensor * rmsnorm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}
int t5_bucket(int rel, int num_buckets, int max_dist) {
    int ret = 0; num_buckets /= 2;          // bidirectional
    if (rel > 0) ret += num_buckets;
    int n = std::abs(rel);
    int max_exact = num_buckets / 2;
    if (n < max_exact) ret += n;
    else {
        int v = max_exact + (int) (std::log((double) n / max_exact) /
                std::log((double) max_dist / max_exact) * (num_buckets - max_exact));
        ret += std::min(v, num_buckets - 1);
    }
    return ret;
}
}

bool ts_t5::load(const std::string & p) {
    backend = ts_backend_init();  // GPU: CUDA soft_max fixed
    if (!model.load_backend(p, backend)) return false;
    d_model = model.get_i32("t5.d_model", 2048); d_ff = model.get_i32("t5.d_ff", 5120);
    NL = model.get_i32("t5.n_layers", 24); NH = model.get_i32("t5.n_heads", 32);
    hd = model.get_i32("t5.d_kv", 64);
    rel_buckets = model.get_i32("t5.rel_buckets", 32); rel_max_dist = model.get_i32("t5.rel_max_dist", 128);
    eps = model.get_f32("t5.eps", 1e-6f);
    return true;
}

void ts_t5::encode(const int32_t * ids, int n, std::vector<float> & out) {
    auto W = [&](const std::string & s) { return model.need(s); };
    const size_t cs = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> buf(cs); ggml_init_params ip{ cs, buf.data(), true };
    ggml_context * c = ggml_init(ip);

    ggml_tensor * tid = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);        ggml_set_input(tid);
    ggml_tensor * bkt = ggml_new_tensor_1d(c, GGML_TYPE_I32, n * n);    ggml_set_input(bkt); // bucket ids, m=q*n+k
    ggml_tensor * pad = ggml_new_tensor_2d(c, GGML_TYPE_F32, n, n);     ggml_set_input(pad); // padding mask [n_kv,n_q]
    ggml_tensor * x = ggml_get_rows(c, W("tok_emb"), tid);             // [d_model, n]

    // relative position bias [n_kv, n_q, NH]
    ggml_tensor * rb = ggml_get_rows(c, W("rel_bias"), bkt);           // [NH, n*n]
    rb = ggml_reshape_3d(c, rb, NH, n, n);                             // [NH, k, q]
    ggml_tensor * bias = ggml_cont(c, ggml_permute(c, rb, 2, 0, 1, 3)); // [k, q, NH]

    for (int i = 0; i < NL; ++i) {
        std::string b = "blk." + std::to_string(i);
        ggml_tensor * res = x;
        ggml_tensor * h = rmsnorm(c, x, W(b + ".attn_norm"), eps);
        ggml_tensor * q = ggml_mul_mat(c, W(b + ".q"), h);
        ggml_tensor * k = ggml_mul_mat(c, W(b + ".k"), h);
        ggml_tensor * v = ggml_mul_mat(c, W(b + ".v"), h);
        q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, q, hd, NH, n), 0, 2, 1, 3)); // [hd,n,NH]
        k = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, k, hd, NH, n), 0, 2, 1, 3));
        v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, hd, NH, n), 0, 2, 1, 3));
        // T5 runs on CPU (per-head relative bias is incompatible with this build's CUDA
        // flash-attn, and standalone CUDA soft_max is broken): plain soft_max, no logit scale.
        ggml_tensor * sc = ggml_mul_mat(c, k, q);                       // [n_kv,n_q,NH]
        sc = ggml_add(c, sc, bias);
        sc = ggml_add(c, sc, pad);
        sc = ggml_soft_max(c, sc);
        ggml_tensor * vt = ggml_cont(c, ggml_transpose(c, v));          // [n_kv,hd,NH]
        ggml_tensor * o = ggml_mul_mat(c, vt, sc);                      // [hd,n_q,NH]
        o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));              // [hd,NH,n]
        o = ggml_reshape_2d(c, o, d_model, n);
        o = ggml_mul_mat(c, W(b + ".o"), o);
        x = ggml_add(c, res, o);
        // gated-gelu FFN
        res = x;
        h = rmsnorm(c, x, W(b + ".ffn_norm"), eps);
        ggml_tensor * g0 = ggml_gelu(c, ggml_mul_mat(c, W(b + ".wi_0"), h));
        ggml_tensor * g1 = ggml_mul_mat(c, W(b + ".wi_1"), h);
        ggml_tensor * ff = ggml_mul_mat(c, W(b + ".wo"), ggml_mul(c, g0, g1));
        x = ggml_add(c, res, ff);
    }
    x = rmsnorm(c, x, W("out_norm"), eps);
    ggml_set_name(x, "t5_out"); ggml_set_output(x);

    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(gf, x);
    ggml_backend_t be = backend;
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al, gf);
    std::vector<int32_t> bk(n * n);
    for (int q = 0; q < n; ++q) for (int kk = 0; kk < n; ++kk) bk[q * n + kk] = t5_bucket(kk - q, rel_buckets, rel_max_dist);
    std::vector<float> pd((size_t) n * n);   // pad id = 0
    for (int q = 0; q < n; ++q) for (int kk = 0; kk < n; ++kk) pd[(size_t) q * n + kk] = (ids[kk] != 0) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(tid, ids, 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(bkt, bk.data(), 0, bk.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pad, pd.data(), 0, pd.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);

    out.resize((size_t) d_model * n);
    ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(al); ggml_free(c);
}
