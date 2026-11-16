// =============================================================================
// Types shared by the whole application.
// =============================================================================
#pragma once

#include <stdint.h>

constexpr int N_HORIZONS = 6;           // forecast horizons: +1 .. +6 h

// Result of one model variant.
struct Forecast {
    bool    ok = false;                 // false: model missing or failed
    float   kW[N_HORIZONS] = {};        // predicted PV power per horizon
    int64_t us_per_run = 0;             // mean Invoke() latency
};
