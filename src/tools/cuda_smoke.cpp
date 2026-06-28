// ts-cuda_smoke — minimal soft_max-on-GPU repro across shapes.
#include "common/ts_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <vector>

static bool try_softmax(ggml_backend_t be, int ne0, int ne1, int ne2) {
    size_t cs = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    std::vector<uint8_t> buf(cs);
    ggml_init_params ip{ cs, buf.data(), true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_input(x);
    ggml_tensor * y = ggml_soft_max(c, x);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    bool ok = ggml_gallocr_alloc_graph(al, gf);
    std::vector<float> d((size_t) ne0 * ne1 * ne2, 0.1f);
    ggml_backend_tensor_set(x, d.data(), 0, d.size() * sizeof(float));
    ggml_status st = ggml_backend_graph_compute(be, gf);
    printf("soft_max [%d,%d,%d] alloc=%d status=%d\n", ne0, ne1, ne2, ok, (int) st);
    ggml_gallocr_free(al); ggml_free(c);
    return st == GGML_STATUS_SUCCESS;
}

static bool try_flash(ggml_backend_t be, int hd, int nkv, int nq, int nh) {
    size_t cs = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    std::vector<uint8_t> buf(cs); ggml_init_params ip{ cs, buf.data(), true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(c, GGML_TYPE_F32, hd, nq, nh, 1); ggml_set_input(q);
    ggml_tensor * k = ggml_new_tensor_4d(c, GGML_TYPE_F16, hd, nkv, nh, 1); ggml_set_input(k);
    ggml_tensor * v = ggml_new_tensor_4d(c, GGML_TYPE_F16, hd, nkv, nh, 1); ggml_set_input(v);
    ggml_tensor * y = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f/8.0f, 0.0f, 0.0f);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(c); ggml_build_forward_expand(gf, y);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    bool ok = ggml_gallocr_alloc_graph(al, gf);
    ggml_status st = ggml_backend_graph_compute(be, gf);
    printf("flash_attn hd=%d nkv=%d nq=%d nh=%d alloc=%d status=%d outne=[%ld,%ld,%ld,%ld]\n",
           hd, nkv, nq, nh, ok, (int) st, (long)y->ne[0],(long)y->ne[1],(long)y->ne[2],(long)y->ne[3]);
    ggml_gallocr_free(al); ggml_free(c);
    return st == GGML_STATUS_SUCCESS;
}

int main() {
    ggml_backend_t be = ts_backend_init();
    try_softmax(be, 128, 10, 1);
    try_flash(be, 64, 77, 77, 8);
    try_flash(be, 64, 420, 420, 16);
    ggml_backend_free(be);
    return 0;
}
