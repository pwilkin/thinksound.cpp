#include "mmdit.h"
#include "common/ts_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cmath>
#include <cstring>

namespace {
const float EPS = 1e-6f;

ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    if (b) y = ggml_add(c, y, b);
    return y;
}

// conv on [T,IC] -> [OC,T_out].  Weight `w` is the mul_mat FIRST operand, so it can be
// bf16/quantized (im2col only uses its shape). f32 im2col (ggml_conv_1d forces F16).
ggml_tensor * conv1d_f32(ggml_context * c, ggml_tensor * w, ggml_tensor * x, int s, int p, int d) {
    ggml_tensor * im = ggml_im2col(c, w, x, s, 0, p, 0, d, 0, false, GGML_TYPE_F32); // [IC*K,OL,1]
    ggml_tensor * cols = ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]);   // [IC*K,OL]
    ggml_tensor * wr   = ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]);       // [IC*K,OC]
    return ggml_mul_mat(c, wr, cols);                                               // [OC,OL]
}

// channels-last conv: canonical [IC,S] -> [OC,S]
ggml_tensor * conv_cl(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                      int s, int p, int d) {
    ggml_tensor * xt = ggml_cont(c, ggml_transpose(c, x));     // [S,IC]
    ggml_tensor * y  = conv1d_f32(c, w, xt, s, p, d);          // [OC,S]  (already canonical)
    if (b) y = ggml_add(c, y, b);                              // b:[OC] broadcasts over S
    return y;
}

ggml_tensor * silu(ggml_context * c, ggml_tensor * x) { return ggml_silu(c, x); }

// SwiGLU conv-MLP (k3,pad1,no bias). prefix.{w1,w2,w3}.weight
ggml_tensor * convmlp(ggml_context * c, ts_model & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * h1 = conv_cl(c, m.need(p + ".w1.weight"), nullptr, x, 1, 1, 1);
    ggml_tensor * h3 = conv_cl(c, m.need(p + ".w3.weight"), nullptr, x, 1, 1, 1);
    ggml_tensor * h  = ggml_mul(c, silu(c, h1), h3);
    return conv_cl(c, m.need(p + ".w2.weight"), nullptr, h, 1, 1, 1);
}

// SwiGLU linear-MLP. prefix.{w1,w2,w3}.weight
ggml_tensor * mlp(ggml_context * c, ts_model & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * h1 = lin(c, m.need(p + ".w1.weight"), nullptr, x);
    ggml_tensor * h3 = lin(c, m.need(p + ".w3.weight"), nullptr, x);
    ggml_tensor * h  = ggml_mul(c, silu(c, h1), h3);
    return lin(c, m.need(p + ".w2.weight"), nullptr, h);
}

// modulate: norm*(1+scale)+shift   (scale,shift broadcast over seq when [H,1])
ggml_tensor * modulate(ggml_context * c, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale, ggml_tensor * one) {
    return ggml_add(c, ggml_mul(c, x, ggml_add(c, scale, one)), shift);
}

ggml_tensor * layernorm(ggml_context * c, ggml_tensor * x) { return ggml_norm(c, x, 1e-5f); }  // nn.LayerNorm default eps (critical: padding tokens have ~0 variance)

// chunk [n*H, S] -> part i [H,S] (cont: CUDA binary ops require contiguous operands)
ggml_tensor * chunk(ggml_context * c, ggml_tensor * t, int H, int i) {
    return ggml_cont(c, ggml_view_2d(c, t, H, t->ne[1], t->nb[1], (size_t) i * H * t->nb[0]));
}

// extract q/k/v from interleaved qkv [3H,S]; returns [head_dim, n_head, S]
void split_qkv(ggml_context * c, ggml_tensor * qkv, int H, int nh, int hd, int S,
               ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v) {
    ggml_tensor * q3 = ggml_reshape_3d(c, qkv, 3, H, S);               // [3,H,S] j-fastest
    auto pick = [&](int j) {
        ggml_tensor * t = ggml_view_3d(c, q3, 1, H, S, q3->nb[1], q3->nb[2], (size_t) j * q3->nb[0]);
        t = ggml_cont(c, t);
        return ggml_reshape_3d(c, t, hd, nh, S);                       // [hd,nh,S]
    };
    *q = pick(0); *k = pick(1); *v = pick(2);
}

ggml_tensor * rmsnorm_w(ggml_context * c, ggml_tensor * x, ggml_tensor * w) {
    return ggml_mul(c, ggml_rms_norm(c, x, EPS), w);
}

// attention over q,k,v [hd,nh,S]; returns [H, S].
// Uses fused flash-attention (the standalone CUDA soft_max kernel is broken in this
// ggml build; flash_attn is the well-tested GPU path and also works on CPU).
ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                        int hd, int nh, int Sq) {
    const float scale = 1.0f / sqrtf((float) hd);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));                   // [hd,S,nh] f32
    k = ggml_cast(c, ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3)), GGML_TYPE_F16);
    v = ggml_cast(c, ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3)), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(c, q, k, v, nullptr, scale, 0.0f, 0.0f); // [hd,nh,Sq,1]
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    return ggml_reshape_2d(c, o, hd * nh, Sq);                          // [H,Sq]
}

ggml_tensor * rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos, int hd, float theta, float fscale) {
    // x [hd,nh,S]; mode NORMAL = interleaved adjacent pairs
    return ggml_rope_ext(c, x, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, theta, fscale,
                         0.0f, 1.0f, 0.0f, 0.0f);
}
} // namespace

ts_dit::~ts_dit() { if (backend) ggml_backend_free(backend); }

bool ts_dit::load(const std::string & gguf_path) {
    backend = ts_backend_init();
    if (!model.load_backend(gguf_path, backend)) return false;
    hidden      = model.get_i32("dit.hidden_dim", 1024);
    n_head      = model.get_i32("dit.num_heads", 16);
    head_dim    = hidden / n_head;
    depth       = model.get_i32("dit.depth", 21);
    fused_depth = model.get_i32("dit.fused_depth", 14);
    latent_dim  = model.get_i32("dit.latent_dim", 64);
    rope_theta  = model.get_f32("dit.rope_theta", 10000.f);
    t_max_period= model.get_f32("dit.t_embed_max_period", 1.f);
    return true;
}

std::vector<float> ts_dit::forward(const float * latent, int64_t T, float t,
                                   const float * clip_f_d, int64_t clip_S,
                                   const float * sync_f_d, int64_t sync_S,
                                   const float * text_f_d,
                                   const float * t5_d,
                                   const float * global_text_d,
                                   std::map<std::string, std::vector<float>> * dbg) {
    ts_model & M = model;
    auto W = [&](const std::string & n) { return M.need(n); };
    const int H = hidden, nh = n_head, hd = head_dim;
    const int n_joint = depth - fused_depth;          // 7
    const int text_S = (int) text_seq;                 // 77

    const size_t ctx_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    std::vector<uint8_t> buf(ctx_size);
    ggml_init_params ip{ ctx_size, buf.data(), true };
    ggml_context * c = ggml_init(ip);

    // ---- inputs ----
    ggml_tensor * in_lat   = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, latent_dim);   ggml_set_input(in_lat);
    ggml_tensor * in_clip  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1024, clip_S);    ggml_set_input(in_clip);
    ggml_tensor * in_sync  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 768, sync_S);     ggml_set_input(in_sync);
    ggml_tensor * in_text  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1024, text_S);    ggml_set_input(in_text);
    ggml_tensor * in_t5    = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, text_S);    ggml_set_input(in_t5);
    ggml_tensor * in_gtext = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);            ggml_set_input(in_gtext);
    ggml_tensor * in_temb  = ggml_new_tensor_1d(c, GGML_TYPE_F32, H);               ggml_set_input(in_temb); // freq emb
    ggml_tensor * one      = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);               ggml_set_input(one);
    ggml_tensor * pos_lat  = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);               ggml_set_input(pos_lat);
    ggml_tensor * pos_clip = ggml_new_tensor_1d(c, GGML_TYPE_I32, clip_S);          ggml_set_input(pos_clip);
    ggml_tensor * sync_idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);               ggml_set_input(sync_idx); // nearest-exact map

    // ---- audio_input_proj: [64,T] -> [H,T] ----
    ggml_tensor * lat = ggml_cont(c, ggml_transpose(c, in_lat));      // [64,T]
    lat = conv_cl(c, W("audio_input_proj.0.weight"), W("audio_input_proj.0.bias"), lat, 1, 1, 1);
    lat = silu(c, lat);
    lat = convmlp(c, M, "audio_input_proj.2", lat);                   // [H,T]
    ggml_tensor * x_in = lat;  ggml_set_name(x_in, "x_in"); ggml_set_output(x_in);

    // ---- t_embed mlp: freq[H] -> [H] ----
    ggml_tensor * te = lin(c, W("t_embed.mlp.0.weight"), W("t_embed.mlp.0.bias"), in_temb);
    te = silu(c, te);
    te = lin(c, W("t_embed.mlp.2.weight"), W("t_embed.mlp.2.bias"), te);            // [H]
    ggml_set_name(te, "t_emb"); ggml_set_output(te);

    // ---- conditioning preprocess ----
    // clip: linear+silu+convmlp
    ggml_tensor * clip = lin(c, W("clip_input_proj.0.weight"), W("clip_input_proj.0.bias"), in_clip);
    clip = silu(c, clip);
    clip = convmlp(c, M, "clip_input_proj.2", clip);                  // [H,clip_S]
    // sync: + pos_emb (per 8-frame segment), convk7+silu+convmlp, interp -> T
    int n_seg = (int) sync_S / 8;
    ggml_tensor * sync = ggml_reshape_3d(c, in_sync, 768, 8, n_seg);
    ggml_tensor * spe  = ggml_reshape_3d(c, W("sync_pos_emb"), 768, 8, 1);
    sync = ggml_add(c, sync, spe);
    sync = ggml_reshape_2d(c, sync, 768, sync_S);
    sync = conv_cl(c, W("sync_input_proj.0.weight"), W("sync_input_proj.0.bias"), sync, 1, 3, 1);
    sync = silu(c, sync);
    sync = convmlp(c, M, "sync_input_proj.2", sync);                  // [H,sync_S]
    sync = ggml_get_rows(c, sync, sync_idx);                          // nearest-exact -> [H,T]
    // text: pad text_f to 2048, concat t5 along seq, linear+silu+mlp
    ggml_tensor * text_pad = ggml_pad(c, in_text, 1024, 0, 0, 0);     // [2048,77]
    ggml_tensor * text_cat = ggml_concat(c, text_pad, in_t5, 1);      // [2048,154]
    ggml_tensor * text = lin(c, W("text_input_proj.0.weight"), W("text_input_proj.0.bias"), text_cat);
    text = silu(c, text);
    text = mlp(c, M, "text_input_proj.2", text);                     // [H,154]
    // conditioning vectors
    ggml_tensor * text_fc = lin(c, W("text_cond_proj.weight"), W("text_cond_proj.bias"), in_gtext); // [H]
    ggml_tensor * clip_mean = ggml_mean(c, ggml_cont(c, ggml_transpose(c, clip)));  // [1,H]
    clip_mean = ggml_reshape_1d(c, clip_mean, H);
    ggml_tensor * clip_fc = lin(c, W("clip_cond_proj.weight"), W("clip_cond_proj.bias"), clip_mean); // [H]
    // global_c = t_emb + global_cond_mlp(clip_fc + text_fc)
    ggml_set_name(clip_fc, "clip_fc"); ggml_set_output(clip_fc);
    ggml_set_name(text_fc, "text_fc"); ggml_set_output(text_fc);
    ggml_tensor * gc = mlp(c, M, "global_cond_mlp", ggml_add(c, clip_fc, text_fc)); // [H]
    ggml_set_name(gc, "gcmlp"); ggml_set_output(gc);
    gc = ggml_add(c, gc, te);                                          // [H]
    ggml_tensor * global_c = ggml_reshape_2d(c, gc, H, 1);             // [H,1]
    ggml_tensor * extended_c = ggml_add(c, sync, global_c);           // [H,T] (global_c broadcast)
    ggml_set_name(clip, "clip_pp"); ggml_set_output(clip);
    ggml_set_name(text, "text_pp"); ggml_set_output(text);
    ggml_set_name(global_c, "global_c"); ggml_set_output(global_c);
    ggml_set_name(extended_c, "extended_c"); ggml_set_output(extended_c);

    // ---- joint blocks (7) ----
    auto pre_attn = [&](const std::string & bp, ggml_tensor * x, ggml_tensor * cond, ggml_tensor * pos,
                        float fscale, bool is_text, bool pre_only,
                        ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v,
                        ggml_tensor ** gate_msa, ggml_tensor ** shift_mlp, ggml_tensor ** scale_mlp, ggml_tensor ** gate_mlp,
                        ggml_tensor ** x_mod_out) {
        int S = (int) x->ne[1];
        ggml_tensor * mod = lin(c, W(bp + ".adaLN_modulation.1.weight"), W(bp + ".adaLN_modulation.1.bias"), silu(c, cond));
        ggml_tensor * shift_msa = chunk(c, mod, H, 0);
        ggml_tensor * scale_msa = chunk(c, mod, H, 1);
        if (!pre_only) {
            *gate_msa  = chunk(c, mod, H, 2);
            *shift_mlp = chunk(c, mod, H, 3);
            *scale_mlp = chunk(c, mod, H, 4);
            *gate_mlp  = chunk(c, mod, H, 5);
        }
        ggml_tensor * xn = modulate(c, layernorm(c, x), shift_msa, scale_msa, one);
        ggml_tensor * qkv = lin(c, W(bp + ".attn.qkv.weight"), W(bp + ".attn.qkv.bias"), xn);
        split_qkv(c, qkv, H, nh, hd, S, q, k, v);
        *q = rmsnorm_w(c, *q, W(bp + ".attn.q_norm.weight"));
        *k = rmsnorm_w(c, *k, W(bp + ".attn.k_norm.weight"));
        if (!is_text) { *q = rope(c, *q, pos, hd, rope_theta, fscale); *k = rope(c, *k, pos, hd, rope_theta, fscale); }
        *x_mod_out = xn;
    };
    auto post_attn = [&](const std::string & bp, ggml_tensor * x, ggml_tensor * attn_out,
                         ggml_tensor * gate_msa, ggml_tensor * shift_mlp, ggml_tensor * scale_mlp, ggml_tensor * gate_mlp,
                         bool is_text) {
        ggml_tensor * l1 = is_text ? lin(c, W(bp + ".linear1.weight"), W(bp + ".linear1.bias"), attn_out)
                                   : conv_cl(c, W(bp + ".linear1.weight"), W(bp + ".linear1.bias"), attn_out, 1, 1, 1);
        x = ggml_add(c, x, ggml_mul(c, l1, gate_msa));
        ggml_tensor * r = modulate(c, layernorm(c, x), shift_mlp, scale_mlp, one);
        ggml_tensor * f = is_text ? mlp(c, M, bp + ".ffn", r) : convmlp(c, M, bp + ".ffn", r);
        return ggml_add(c, x, ggml_mul(c, f, gate_mlp));
    };

    float clip_fscale = (float) T / (float) clip_S;
    for (int i = 0; i < n_joint; ++i) {
        std::string jb = "joint_blocks." + std::to_string(i);
        bool pre_only = (i == n_joint - 1);
        ggml_tensor *ql,*kl,*vl,*qc,*kc,*vc,*qt,*kt,*vt;
        ggml_tensor *gml,*sfl,*scl,*gpl,*xml;  // latent gates
        ggml_tensor *gmc,*sfc,*scc,*gpc,*xmc;  // clip gates
        ggml_tensor *gmt,*sft,*sct,*gpt,*xmt;  // text gates
        pre_attn(jb + ".latent_block", lat,  extended_c, pos_lat,  1.0f,        false, false, &ql,&kl,&vl,&gml,&sfl,&scl,&gpl,&xml);
        pre_attn(jb + ".clip_block",   clip, global_c,   pos_clip, clip_fscale, false, pre_only, &qc,&kc,&vc,&gmc,&sfc,&scc,&gpc,&xmc);
        pre_attn(jb + ".text_block",   text, global_c,   nullptr,  1.0f,        true,  pre_only, &qt,&kt,&vt,&gmt,&sft,&sct,&gpt,&xmt);
        if (i == 0) { ggml_set_name(xmt,"text_xn"); ggml_set_output(xmt); ggml_set_name(vt,"text_v"); ggml_set_output(vt); }
        // joint attention: concat along seq (ne[2])
        int Sl = (int) lat->ne[1], Sc = (int) clip->ne[1], St = (int) text->ne[1];
        ggml_tensor * Q = ggml_concat(c, ggml_concat(c, ql, qc, 2), qt, 2);
        ggml_tensor * K = ggml_concat(c, ggml_concat(c, kl, kc, 2), kt, 2);
        ggml_tensor * V = ggml_concat(c, ggml_concat(c, vl, vc, 2), vt, 2);
        int Stot = Sl + Sc + St;
        if (i == 0) { ggml_set_name(Q,"a0_q"); ggml_set_output(Q); ggml_set_name(K,"a0_k"); ggml_set_output(K); ggml_set_name(V,"a0_v"); ggml_set_output(V); }
        ggml_tensor * ao = attention(c, Q, K, V, hd, nh, Stot);        // [H,Stot]
        if (i == 0) { ggml_set_name(ao,"a0_out"); ggml_set_output(ao); }
        ggml_tensor * ao_l = ggml_cont(c, ggml_view_2d(c, ao, H, Sl, ao->nb[1], 0));
        ggml_tensor * ao_c = ggml_cont(c, ggml_view_2d(c, ao, H, Sc, ao->nb[1], (size_t) Sl * ao->nb[1]));
        ggml_tensor * ao_t = ggml_cont(c, ggml_view_2d(c, ao, H, St, ao->nb[1], (size_t)(Sl + Sc) * ao->nb[1]));
        lat = post_attn(jb + ".latent_block", lat, ao_l, gml,sfl,scl,gpl, false);
        if (!pre_only) {
            clip = post_attn(jb + ".clip_block", clip, ao_c, gmc,sfc,scc,gpc, false);
            text = post_attn(jb + ".text_block", text, ao_t, gmt,sft,sct,gpt, true);
        }
        if (i == 0) {
            ggml_set_name(lat, "jb0_lat");   ggml_set_output(lat);
            ggml_set_name(clip, "jb0_clip"); ggml_set_output(clip);
            ggml_set_name(text, "jb0_text"); ggml_set_output(text);
        }
    }
    ggml_set_name(lat, "after_joint"); ggml_set_output(lat);

    // ---- fused blocks (14): single-stream on latent + extended_c ----
    for (int i = 0; i < fused_depth; ++i) {
        std::string fb = "fused_blocks." + std::to_string(i);
        ggml_tensor *q,*k,*v,*gm,*sf,*sc,*gp,*xm;
        pre_attn(fb, lat, extended_c, pos_lat, 1.0f, false, false, &q,&k,&v,&gm,&sf,&sc,&gp,&xm);
        ggml_tensor * ao = attention(c, q, k, v, hd, nh, (int) lat->ne[1]);
        lat = post_attn(fb, lat, ao, gm,sf,sc,gp, false);
    }
    ggml_set_name(lat, "after_fused"); ggml_set_output(lat);

    // ---- final layer: adaLN(extended_c) -> modulate -> conv k7 -> [latent_dim,T] ----
    ggml_tensor * fmod = lin(c, W("final_layer.adaLN_modulation.1.weight"), W("final_layer.adaLN_modulation.1.bias"), silu(c, extended_c));
    ggml_tensor * fsh = chunk(c, fmod, H, 0);
    ggml_tensor * fsc = chunk(c, fmod, H, 1);
    ggml_tensor * fx = modulate(c, layernorm(c, lat), fsh, fsc, one);
    fx = conv_cl(c, W("final_layer.conv.weight"), W("final_layer.conv.bias"), fx, 1, 3, 1);  // [latent_dim,T]
    ggml_set_name(fx, "flow_cl"); ggml_set_output(fx);

    // ---- build + alloc + compute ----
    ggml_cgraph * gf = ggml_new_graph_custom(c, 16384, false);
    ggml_build_forward_expand(gf, fx);
    // ensure stage outputs are in the graph
    ggml_build_forward_expand(gf, x_in); ggml_build_forward_expand(gf, te);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "dit: alloc failed\n"); return {}; }

    // ---- set inputs ----
    auto setT = [&](ggml_tensor * t_, const float * d) { ggml_backend_tensor_set(t_, d, 0, ggml_nbytes(t_)); };
    setT(in_lat, latent); setT(in_clip, clip_f_d); setT(in_sync, sync_f_d);
    setT(in_text, text_f_d); setT(in_t5, t5_d); setT(in_gtext, global_text_d);
    // timestep freq embedding (v2: freq_size=H, max_period=1): emb=[cos(t*f); sin(t*f)]
    std::vector<float> temb(H);
    int half = H / 2;
    float fscale_t = 10000.0f / t_max_period;
    for (int kk = 0; kk < half; ++kk) {
        float fr = fscale_t * powf(10000.0f, -2.0f * kk / (float) H);
        temb[kk]        = cosf(t * fr);
        temb[half + kk] = sinf(t * fr);
    }
    setT(in_temb, temb.data());
    float one_v = 1.0f; ggml_backend_tensor_set(one, &one_v, 0, sizeof(float));
    std::vector<int32_t> pl(T), pc(clip_S), si(T);
    for (int64_t i = 0; i < T; ++i) pl[i] = (int32_t) i;
    for (int64_t i = 0; i < clip_S; ++i) pc[i] = (int32_t) i;
    for (int64_t i = 0; i < T; ++i) si[i] = (int32_t) floorf(((float) i + 0.5f) * (float) sync_S / (float) T); // nearest-exact
    ggml_backend_tensor_set(pos_lat, pl.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(pos_clip, pc.data(), 0, clip_S * sizeof(int32_t));
    ggml_backend_tensor_set(sync_idx, si.data(), 0, T * sizeof(int32_t));

    fprintf(stderr, "dit: graph %d nodes, computing (t=%.4f)...\n", ggml_graph_n_nodes(gf), t);
    ggml_backend_graph_compute(backend, gf);

    auto grab = [&](ggml_tensor * tt) {
        std::vector<float> out(ggml_nelements(tt));
        ggml_backend_tensor_get(tt, out.data(), 0, ggml_nbytes(tt));
        return out;
    };
    if (dbg) {
        (*dbg)["x_in"] = grab(x_in);
        (*dbg)["t_emb"] = grab(te);
        for (const char * nm : {"clip_pp","text_pp","clip_fc","text_fc","gcmlp","global_c","extended_c","text_xn","a0_q","a0_k","a0_v","a0_out","jb0_lat","jb0_clip","jb0_text","after_joint","after_fused"})
            (*dbg)[nm] = grab(ggml_graph_get_tensor(gf, nm));
    }
    std::vector<float> flow = grab(fx);   // [latent_dim, T]

    ggml_gallocr_free(alloc); ggml_free(c);
    return flow;
}

std::vector<float> ts_dit::sample(const float * noise, int64_t T,
                                  const float * clip_f, int64_t clip_S,
                                  const float * sync_f, int64_t sync_S,
                                  const float * text_f, const float * t5, const float * global_text,
                                  int steps, float cfg_scale, std::vector<float> * flow0) {
    // uncond conditioning = learned empties
    ggml_tensor * es = model.need("empty_string_feat");   // [1024,77]
    ggml_tensor * et = model.need("empty_t5_feat");        // [2048,77]
    std::vector<float> es_h(ggml_nelements(es)), et_h(ggml_nelements(et)); // copy from backend
    model.get_data(es, es_h.data()); model.get_data(et, et_h.data());
    const float * empty_string = es_h.data();
    const float * empty_t5     = et_h.data();
    std::vector<float> zeros_gt(1024, 0.0f);

    std::vector<float> x(noise, noise + (size_t) T * latent_dim);   // [T,64]
    for (int s = 0; s < steps; ++s) {
        float t_curr = 1.0f - (float) s / steps;                    // linspace(1,0,steps+1)
        float t_prev = 1.0f - (float) (s + 1) / steps;
        float dt = t_prev - t_curr;
        std::vector<float> fc = forward(x.data(), T, t_curr, clip_f, clip_S, sync_f, sync_S, text_f, t5, global_text);
        std::vector<float> fu = forward(x.data(), T, t_curr, clip_f, clip_S, sync_f, sync_S, empty_string, empty_t5, zeros_gt.data());
        // fc,fu are [latent_dim,T] (ne[0]=latent). x is [T,latent].
        for (int64_t cc = 0; cc < latent_dim; ++cc)
            for (int64_t tt = 0; tt < T; ++tt) {
                float fcv = fc[cc + latent_dim * tt], fuv = fu[cc + latent_dim * tt];
                float v = fuv + (fcv - fuv) * cfg_scale;
                if (s == 0 && flow0) { if (flow0->empty()) flow0->resize((size_t) T * latent_dim); (*flow0)[tt + T * cc] = v; }
                x[tt + T * cc] += dt * v;
            }
        fprintf(stderr, "  step %2d/%d  t=%.4f\n", s + 1, steps, t_curr);
    }
    return x;   // [T,latent_dim]
}
