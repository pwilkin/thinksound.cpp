// ts-tok_test <clip-tok.gguf> <t5-tok.gguf> <golden_text.gguf>
//   tokenize the same caption/cot the golden used and compare ids.
#include "tokenizer.h"
#include "common/ts_model.h"
#include <cstdio>
#include <vector>

static void show(const char * nm, const std::vector<int32_t> & mine, const ggml_tensor * g) {
    const int32_t * b = (const int32_t *) g->data;
    int n = (int) g->ne[0], mism = 0, nz = 0;
    for (int i = 0; i < n; ++i) { if ((int) mine.size() > i && mine[i] != b[i]) mism++; if (b[i] != 0 && b[i] != 49407) nz++; }
    printf("%s: %d ids, mismatches=%d\n  mine: ", nm, n, mism);
    for (int i = 0; i < 12; ++i) printf("%d ", (int) mine.size() > i ? mine[i] : -1);
    printf("\n  gold: "); for (int i = 0; i < 12; ++i) printf("%d ", b[i]);
    printf("\n  -> %s\n", mism == 0 ? "EXACT MATCH" : "**");
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: ts-tok_test <clip-tok> <t5-tok> <golden_text>\n"); return 1; }
    const char * caption = argc > 4 ? argv[4] : "a dog barking";
    const char * cot = argc > 5 ? argv[5] : "A dog barks several times, sharp and clear, in a quiet room.";
    ts_clip_tokenizer clip; ts_t5_tokenizer t5;
    if (!clip.load(argv[1]) || !t5.load(argv[2])) return 1;
    ts_model g; if (!g.load(argv[3])) return 1;

    auto ci = clip.encode(caption, 77);
    auto ti = t5.encode(cot, 77);
    show("CLIP", ci, g.need("clip_input_ids"));
    show("T5",   ti, g.need("t5_input_ids"));
    return 0;
}
