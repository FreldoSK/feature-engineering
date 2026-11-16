// =============================================================================
// The three model variants compared by the application:
//
//   A  raw channels only                 (7 inputs)
//   B  A + constructed features          (17 inputs)
//   C  B + mutual-information selection  (8 inputs)
// =============================================================================
#pragma once

#include "forecast.h"
#include "pv_features.h"

enum class Variant { A, B, C };

constexpr int N_VARIANTS = 3;

// Builds the variant's features from the window and runs its model.
// Returns a Forecast with ok == false if the model is missing or fails.
Forecast run_variant(Variant v, const window_t &w);
