#include "clip_text.h"
#include "common/ts_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cmath>

ts_clip_text::~ts_clip_text() { if (backend) ggml_backend_free(backend); }

namespace {
const float EPS = 1e-5f;
ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(c, w, x); if (b) y = ggml_add(c, y, b); return y;
}
ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, EPS), w), b);
}
}

bool ts_clip_text::load(const std::string & p) {
    backend = ts_backend_init();
    if (!model.load_backend(p, backend)) return false;
    H = model.get_i32("clip.hidden", 1024); NL = model.get_i32("clip.layers", 24);
    NH = model.get_i32("clip.heads", 16); hd = H / NH; ctx = model.get_i32("clip.ctx", 77);
    eos_id = model.get_i32("clip.eos_id", 49407);
    return true;
}

void ts_clip_text::encode(const int32_t * ids, int n, std::vector<float> & per_token, std::vector<float> & global) {
    auto W = [&](const std::string & s) { return model.need(s); };
    const size_t cs = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> buf(cs); ggml_init_params ip{ cs, buf.data(), true };
    ggml_context * c = ggml_init(ip);

    ggml_tensor * tid  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(tid);
    ggml_tensor * pid  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pid);
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n, n); ggml_set_input(mask); // causal+padding [n_kv,n_q]
    ggml_tensor * x = ggml_add(c, ggml_get_rows(c, W("tok_emb"), tid), ggml_get_rows(c, W("pos_emb"), pid)); // [H,n]

    for (int i = 0; i < NL; ++i) {
        std::string b = "blk." + std::to_string(i);
        ggml_tensor * res = x;
        ggml_tensor * h = ln(c, x, W(b + ".ln1.w"), W(b + ".ln1.b"));
        ggml_tensor * q = lin(c, W(b + ".q.w"), W(b + ".q.b"), h);
        ggml_tensor * k = lin(c, W(b + ".k.w"), W(b + ".k.b"), h);
        ggml_tensor * v = lin(c, W(b + ".v.w"), W(b + ".v.b"), h);
        q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, q, hd, NH, n), 0, 2, 1, 3)); // [hd,n,NH]
        k = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, k, hd, NH, n), 0, 2, 1, 3));
        v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, hd, NH, n), 0, 2, 1, 3));
        // CLIP runs on CPU (consistent with T5): standard scaled soft_max with causal+pad mask.
        ggml_tensor * sc = ggml_soft_max_ext(c, ggml_mul_mat(c, k, q), mask, 1.0f / sqrtf((float) hd), 0.0f); // [n_kv,n_q,NH]
        ggml_tensor * vt = ggml_cont(c, ggml_transpose(c, v));                              // [n_kv,hd,NH]
        ggml_tensor * o = ggml_mul_mat(c, vt, sc);                                          // [hd,n_q,NH]
        o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));                                   // [hd,NH,n]
        o = ggml_reshape_2d(c, o, H, n);
        o = lin(c, W(b + ".out.w"), W(b + ".out.b"), o);
        x = ggml_add(c, res, o);
        res = x;
        h = ln(c, x, W(b + ".ln2.w"), W(b + ".ln2.b"));
        h = lin(c, W(b + ".fc1.w"), W(b + ".fc1.b"), h);
        h = ggml_gelu_quick(c, h);
        h = lin(c, W(b + ".fc2.w"), W(b + ".fc2.b"), h);
        x = ggml_add(c, res, h);
    }
    x = ln(c, x, W("out_ln.w"), W("out_ln.b"));    // [H,n] last_hidden_state
    ggml_set_name(x, "per_token"); ggml_set_output(x);

    // pooled @ eos position (first argmax of ids) -> text_projection
    int eos_pos = 0, mx = ids[0];
    for (int i = 1; i < n; ++i) if (ids[i] > mx) { mx = ids[i]; eos_pos = i; }
    ggml_tensor * pooled = ggml_cont(c, ggml_view_1d(c, x, H, (size_t) eos_pos * x->nb[1]));
    ggml_tensor * g = ggml_mul_mat(c, W("text_proj"), pooled);   // [H]
    ggml_set_name(g, "global"); ggml_set_output(g);

    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(gf, x); ggml_build_forward_expand(gf, g);
    ggml_backend_t be = backend;
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al, gf);
    std::vector<int32_t> pos(n); for (int i = 0; i < n; ++i) pos[i] = i;
    // causal + padding mask: key k attendable from query q iff k<=q (causal) and k<=eos_pos (not padding)
    std::vector<float> mdat((size_t) n * n);
    for (int q = 0; q < n; ++q)
        for (int kk = 0; kk < n; ++kk)
            mdat[(size_t) q * n + kk] = (kk <= q && kk <= eos_pos) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(tid, ids, 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(pid, pos.data(), 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(mask, mdat.data(), 0, mdat.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);

    per_token.resize((size_t) H * n); ggml_backend_tensor_get(x, per_token.data(), 0, per_token.size() * sizeof(float));
    global.resize(H);                 ggml_backend_tensor_get(g, global.data(), 0, global.size() * sizeof(float));
    ggml_gallocr_free(al); ggml_free(c);
}
