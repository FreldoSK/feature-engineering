// =============================================================================
// Pulls in the auto-generated model headers.
//
// The model headers define their weights as static arrays, so this file must
// be included by exactly ONE .cpp (variants.cpp) -- otherwise every model
// ends up in flash more than once.
// =============================================================================
#pragma once

#include "pv_features.h"

// ---- variant A (optional) --------------------------------------------------
#if __has_include("model_A.h")
#include "model_A.h"
#define HAVE_MODEL_A 1
#else
#define HAVE_MODEL_A 0
#endif

// ---- variant B (optional) --------------------------------------------------
#if __has_include("model_B.h")
#include "model_B.h"
#define HAVE_MODEL_B 1
#else
#define HAVE_MODEL_B 0
#endif

// ---- variant C -------------------------------------------------------------
// model_C.h if present, otherwise model_data.h (also variant C, with
// 8 MI-selected features) mapped onto the MODEL_C_* names.
#if __has_include("model_C.h")
#include "model_C.h"
#else
#include "model_data.h"

#ifndef MODEL_N_OUTPUTS
#define MODEL_N_OUTPUTS 1       // model_data.h omits it; its output is [1, 1]
#endif

#define MODEL_C_N_FEATURES  MODEL_N_FEATURES
#define MODEL_C_N_OUTPUTS   MODEL_N_OUTPUTS
#define MODEL_C_IN_SCALE    MODEL_IN_SCALE
#define MODEL_C_IN_ZERO     MODEL_IN_ZERO
#define MODEL_C_OUT_SCALE   MODEL_OUT_SCALE
#define MODEL_C_OUT_ZERO    MODEL_OUT_ZERO
#define FEAT_SCALE_C        FEAT_SCALE
#define FEAT_MIN_C          FEAT_MIN
#define TARGET_SCALE_C      TARGET_SCALE
#define TARGET_MIN_C        TARGET_MIN
#define g_model_C           g_model

// Column order from the model_data.h "Feature order" comment -- keep in sync
// whenever model_data.h is regenerated.
static const int FEAT_INDEX_C[MODEL_C_N_FEATURES] = {
    FB_GHI,           // GHI[kW1m2]
    FB_POA,           // POA Irr[kW1m2]
    FB_HOUR_SIN,      // hour_sin
    FB_HOUR_COS,      // hour_cos
    FX_HOUR_SIN_FUT,  // hour_sin_future
    FX_HOUR_COS_FUT,  // hour_cos_future
    FB_POA_GHI,       // poa_ghi
    FB_PS_MEAN6,      // P_Solar[kW]_mean6
};
#endif
