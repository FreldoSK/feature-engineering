// =============================================================================
// Runs one int8-quantized TFLite Micro model on prepared float features.
//
// All models share one tensor arena; each interpreter lives only for the
// duration of run_model(), so only one is allocated at a time.
// =============================================================================
#pragma once

#include "forecast.h"

struct QuantParams {
    float scale;
    int   zero_point;
};

// Everything needed to feed a model and decode its output.
// Filled from the auto-generated model_X.h headers (see variants.cpp).
struct ModelSpec {
    const char          *name;
    const unsigned char *flatbuffer;
    int                  n_features;    // input columns per time step
    int                  n_outputs;     // values in the output tensor
    const float         *feat_scale;    // min-max scaler: v * scale + min
    const float         *feat_min;
    QuantParams          input;
    QuantParams          output;
    float                target_scale;  // inverse scaler for the output
    float                target_min;
};

// Runs the model n_runs times.
//   features  n_runs input windows back to back, each [WIN_ROWS][n_features]
//   result    receives n_runs * n_outputs values (must fit N_HORIZONS)
bool run_model(const ModelSpec &spec, const float *features, int n_runs,
               Forecast &result);
