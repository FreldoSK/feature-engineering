#include "variants.h"

#include <stdio.h>

#include "model_runner.h"
#include "models.h"             // included ONLY here -- see models.h

namespace {

// ---- variant C run count ---------------------------------------------------
// A single-output model C is run once per horizon h = 1..6, with
// hour_sin_future / hour_cos_future set to t + h. A 6-output model C runs
// once (shift 1 is then only a placeholder for those two columns).
static_assert(MODEL_C_N_OUTPUTS == 1 || MODEL_C_N_OUTPUTS == N_HORIZONS,
              "model C must have 1 or N_HORIZONS outputs");
constexpr int RUNS_C = (MODEL_C_N_OUTPUTS == 1) ? N_HORIZONS : 1;

// ---- shared feature buffer, big enough for every variant -------------------
constexpr int max_int(int a, int b) { return a > b ? a : b; }

constexpr int FEATURE_BUF_SIZE =
    WIN_ROWS * max_int(max_int(N_FEAT_A, N_FEAT_B), RUNS_C * MODEL_C_N_FEATURES);

float features[FEATURE_BUF_SIZE];

// ---- per-variant model specs -----------------------------------------------
#if HAVE_MODEL_A
const ModelSpec SPEC_A = {
    "A", g_model_A, MODEL_A_N_FEATURES, N_HORIZONS,
    FEAT_SCALE_A, FEAT_MIN_A,
    { MODEL_A_IN_SCALE,  MODEL_A_IN_ZERO  },
    { MODEL_A_OUT_SCALE, MODEL_A_OUT_ZERO },
    TARGET_SCALE_A, TARGET_MIN_A,
};
static_assert(MODEL_A_N_FEATURES == N_FEAT_A, "model A expects the raw channels");
#endif

#if HAVE_MODEL_B
const ModelSpec SPEC_B = {
    "B", g_model_B, MODEL_B_N_FEATURES, N_HORIZONS,
    FEAT_SCALE_B, FEAT_MIN_B,
    { MODEL_B_IN_SCALE,  MODEL_B_IN_ZERO  },
    { MODEL_B_OUT_SCALE, MODEL_B_OUT_ZERO },
    TARGET_SCALE_B, TARGET_MIN_B,
};
static_assert(MODEL_B_N_FEATURES == N_FEAT_B, "model B expects all B features");
#endif

const ModelSpec SPEC_C = {
    "C", g_model_C, MODEL_C_N_FEATURES, MODEL_C_N_OUTPUTS,
    FEAT_SCALE_C, FEAT_MIN_C,
    { MODEL_C_IN_SCALE,  MODEL_C_IN_ZERO  },
    { MODEL_C_OUT_SCALE, MODEL_C_OUT_ZERO },
    TARGET_SCALE_C, TARGET_MIN_C,
};

// ---- runners ---------------------------------------------------------------
[[maybe_unused]] Forecast skipped(const char *name)
{
    printf("%s: model_%s.h not present, skipped\n", name, name);
    return Forecast{};
}

Forecast failed(const char *what)
{
    printf("%s failed\n", what);
    return Forecast{};
}

Forecast run_A(const window_t &w)
{
#if HAVE_MODEL_A
    auto rows = reinterpret_cast<float (*)[N_FEAT_A]>(features);
    if (features_A(&w, rows) != 0)
        return failed("features_A");

    Forecast f;
    run_model(SPEC_A, features, 1, f);
    return f;
#else
    (void)w;
    return skipped("A");
#endif
}

Forecast run_B(const window_t &w)
{
#if HAVE_MODEL_B
    auto rows = reinterpret_cast<float (*)[N_FEAT_B]>(features);
    if (features_B(&w, rows) != 0)
        return failed("features_B");

    Forecast f;
    run_model(SPEC_B, features, 1, f);
    return f;
#else
    (void)w;
    return skipped("B");
#endif
}

Forecast run_C(const window_t &w)
{
    const int window_size = WIN_ROWS * MODEL_C_N_FEATURES;
    for (int r = 0; r < RUNS_C; r++) {
        float hour_shift = (float)(r + 1);
        if (features_C(&w, FEAT_INDEX_C, MODEL_C_N_FEATURES, hour_shift,
                       features + r * window_size) != 0)
            return failed("features_C");
    }

    Forecast f;
    run_model(SPEC_C, features, RUNS_C, f);
    return f;
}

}  // namespace

Forecast run_variant(Variant v, const window_t &w)
{
    switch (v) {
        case Variant::A: return run_A(w);
        case Variant::B: return run_B(w);
        case Variant::C: return run_C(w);
    }
    return Forecast{};
}
