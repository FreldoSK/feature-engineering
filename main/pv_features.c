#include "pv_features.h"

#include <math.h>
#include <string.h>

static const float TWO_PI = 6.283185307179586f;

// ---- helpers ---------------------------------------------------------------

// Mean of channel ch over raw[i-ROLL .. i-1] == shift(1).rolling(6).mean()
static float roll_mean(const window_t *w, int i, int ch)
{
    float sum = 0.0f;
    for (int k = i - ROLL; k < i; k++)
        sum += w->raw[k][ch];
    return sum / (float)ROLL;
}

// Same window; pandas .std() uses ddof = 1, i.e. divides by ROLL - 1.
static float roll_std(const window_t *w, int i, int ch)
{
    float mean = roll_mean(w, i, ch);
    float sum  = 0.0f;
    for (int k = i - ROLL; k < i; k++) {
        float d = w->raw[k][ch] - mean;
        sum += d * d;
    }
    return sqrtf(sum / (float)(ROLL - 1));
}

// sin / cos encoding of a value with the given period.
static void cyclic(float value, float period, float *out_sin, float *out_cos)
{
    *out_sin = sinf(TWO_PI * value / period);
    *out_cos = cosf(TWO_PI * value / period);
}

// POA / GHI, with GHI == 0 mapped to 0, then clipped to [0, 5].
static float poa_ghi_ratio(float poa, float ghi)
{
    if (ghi == 0.0f)
        return 0.0f;
    float ratio = poa / ghi;
    if (ratio < 0.0f) return 0.0f;
    if (ratio > 5.0f) return 5.0f;
    return ratio;
}

// ---- variant A -------------------------------------------------------------
int features_A(const window_t *w, float out[WIN_ROWS][N_FEAT_A])
{
    if (w->n < WIN_ROWS)
        return -1;

    int first = w->n - WIN_ROWS;
    for (int t = 0; t < WIN_ROWS; t++)
        memcpy(out[t], w->raw[first + t], sizeof out[t]);
    return 0;
}

// ---- variant B -------------------------------------------------------------
int features_B(const window_t *w, float out[WIN_ROWS][N_FEAT_B])
{
    if (w->n < MIN_ROWS)
        return -1;

    int first = w->n - WIN_ROWS;            // first row with valid rolling stats
    for (int t = 0; t < WIN_ROWS; t++) {
        int i = first + t;
        const float *r = w->raw[i];
        float *o = out[t];

        o[FB_P_SOLAR] = r[RAW_P_SOLAR];
        o[FB_GHI]     = r[RAW_GHI];
        o[FB_POA]     = r[RAW_POA];
        o[FB_TEMP]    = r[RAW_TEMP];
        o[FB_HUM]     = r[RAW_HUM];
        o[FB_WS]      = r[RAW_WS];

        cyclic(cal_hour_of_day(&w->time[i]), 24.0f, &o[FB_HOUR_SIN], &o[FB_HOUR_COS]);
        cyclic((float)cal_day_of_year(&w->time[i]), 365.25f, &o[FB_DOY_SIN], &o[FB_DOY_COS]);

        float wd_rad = fmodf(r[RAW_WD], 360.0f) * (float)M_PI / 180.0f;
        o[FB_WD_SIN] = sinf(wd_rad);
        o[FB_WD_COS] = cosf(wd_rad);

        o[FB_POA_GHI] = poa_ghi_ratio(r[RAW_POA], r[RAW_GHI]);

        o[FB_PS_MEAN6]  = roll_mean(w, i, RAW_P_SOLAR);
        o[FB_PS_STD6]   = roll_std (w, i, RAW_P_SOLAR);
        o[FB_GHI_MEAN6] = roll_mean(w, i, RAW_GHI);
        o[FB_GHI_STD6]  = roll_std (w, i, RAW_GHI);
    }
    return 0;
}

// ---- variant C -------------------------------------------------------------
int features_C(const window_t *w, const int *idx, int n_idx,
               float hour_shift, float *out)
{
    for (int j = 0; j < n_idx; j++)
        if (idx[j] < 0 || idx[j] >= N_FEAT_X)
            return -1;

    static float b[WIN_ROWS][N_FEAT_B];
    if (features_B(w, b) != 0)
        return -1;

    int first = w->n - WIN_ROWS;
    for (int t = 0; t < WIN_ROWS; t++) {
        float x[N_FEAT_X];
        memcpy(x, b[t], sizeof b[t]);

        float future_hour = cal_hour_of_day(&w->time[first + t]) + hour_shift;
        cyclic(future_hour, 24.0f, &x[FX_HOUR_SIN_FUT], &x[FX_HOUR_COS_FUT]);

        for (int j = 0; j < n_idx; j++)
            out[t * n_idx + j] = x[idx[j]];
    }
    return 0;
}
