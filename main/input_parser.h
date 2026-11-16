// =============================================================================
// Parses the forecast input window from JSON (cJSON).
//
// Expected keys:
//   "timestamps"      array of "YYYY-MM-DDTHH:MM[:SS]" strings
//   <raw channel>     one number array per channel, same length as timestamps
//   "actual_next_6"   optional ground truth for the next N_HORIZONS hours
// =============================================================================
#pragma once

#include "forecast.h"
#include "pv_features.h"

struct ForecastInput {
    window_t window;
    float    measured[N_HORIZONS];      // valid only if has_measured
    bool     has_measured;
};

// Returns false (and prints the reason) on malformed input.
bool parse_input(const char *json, ForecastInput &in);
