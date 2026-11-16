// =============================================================================
// Console output: input summary and the forecast comparison table.
// =============================================================================
#pragma once

#include "input_parser.h"
#include "variants.h"

void print_input_summary(const window_t &w);

// forecasts[] is indexed by Variant (A, B, C).
void print_report(const ForecastInput &in, const Forecast forecasts[N_VARIANTS]);
