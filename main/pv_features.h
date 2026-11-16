// =============================================================================
// On-device feature engineering for the SOLETE PV forecast.
// Pure C99, no ESP-IDF dependency -- can be unit-tested on a host.
//
// NOTE: deliberately NOT named features.h -- that name collides with
// glibc/newlib <features.h> and makes the preprocessor include this file
// from inside libc headers.
//
// Mirrors features_A() and features_B() from the Colab notebook exactly,
// including pandas' sample standard deviation (ddof = 1).
// =============================================================================
#pragma once

#include "calendar.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WIN_ROWS   24                   // rows fed to the LSTM
#define ROLL       6                    // rolling window length
#define MIN_ROWS   (WIN_ROWS + ROLL)    // 30: shift(1).rolling(6) needs 6 priors
#define MAX_ROWS   48

// ---- raw channel order, as in the JSON and in RAW[] in Colab ----------------
enum {
    RAW_P_SOLAR,
    RAW_GHI,
    RAW_POA,
    RAW_TEMP,
    RAW_HUM,
    RAW_WS,
    RAW_WD,
    N_RAW
};

#define N_FEAT_A  N_RAW                 // variant A = the raw channels

// ---- variant B feature order, as in cols[] in features_B() -----------------
enum {
    FB_P_SOLAR,
    FB_GHI,
    FB_POA,
    FB_TEMP,
    FB_HUM,
    FB_WS,
    FB_HOUR_SIN,
    FB_HOUR_COS,
    FB_DOY_SIN,
    FB_DOY_COS,
    FB_WD_SIN,
    FB_WD_COS,
    FB_POA_GHI,
    FB_PS_MEAN6,
    FB_PS_STD6,
    FB_GHI_MEAN6,
    FB_GHI_STD6,
    N_FEAT_B
};

// ---- extra features, not part of B, selectable by variant C ----------------
// Time of day `hour_shift` hours after the row's timestamp (see features_C).
// Running the model once per shift 1..6 gives a 6-hour forecast -- valid only
// if the model was trained that way.
enum {
    FX_HOUR_SIN_FUT = N_FEAT_B,
    FX_HOUR_COS_FUT,
    N_FEAT_X                            // B + extras: the pool variant C picks from
};

// ---- input window ----------------------------------------------------------
typedef struct {
    int           n;                    // rows supplied, MIN_ROWS..MAX_ROWS
    pv_datetime_t time[MAX_ROWS];
    float         raw[MAX_ROWS][N_RAW];
} window_t;

// All functions return 0 on success, -1 on error.
// Each fills WIN_ROWS rows taken from the end of the window.

// Variant A: the raw channels.
int features_A(const window_t *w, float out[WIN_ROWS][N_FEAT_A]);

// Variant B: constructed features. Needs w->n >= MIN_ROWS.
int features_B(const window_t *w, float out[WIN_ROWS][N_FEAT_B]);

// Variant C: a subset of B + extras, chosen by index.
//   idx[]      FB_* / FX_* values, one per model input column
//   hour_shift hours ahead used for hour_sin_future / hour_cos_future
//   out        [WIN_ROWS][n_idx], row-major
int features_C(const window_t *w, const int *idx, int n_idx,
               float hour_shift, float *out);

#ifdef __cplusplus
}
#endif
