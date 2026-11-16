// =============================================================================
// ESP32-C6 -- PV power forecast, +1 h .. +6 h
//
// Parses the input window from JSON on device, builds the features for all
// three variants in C, and runs all three int8 models on the same window.
//
//   A  raw channels only                 (7 inputs)
//   B  A + constructed features          (17 inputs)
//   C  B + mutual-information selection  (8 inputs)
//
// NOT tested on hardware -- feature construction was verified against pandas
// on a host (max abs difference 5e-07), the rest is a starting point.
//
// Source layout:
//   calendar.*      date arithmetic                         (pure C)
//   pv_features.*   feature engineering for A / B / C       (pure C)
//   input_parser.*  JSON -> window_t
//   fill_op.*       custom TFLM FILL kernel for dynamic-batch LSTMs
//   model_runner.*  quantize -> Invoke -> dequantize
//   models.h        includes the auto-generated model_*.h
//   variants.*      wires features + model for each variant
//   report.*        console output
//
// Build:
//   idf.py add-dependency "espressif/esp-tflite-micro"
//   idf.py set-target esp32c6
//   idf.py build flash monitor
// =============================================================================

#include <stdio.h>

#include "input_parser.h"
#include "report.h"
#include "variants.h"
#include "window_json.h"        // WINDOW_JSON

extern "C" void app_main(void)
{
    static ForecastInput input;

    printf("\n=== PV forecast +1..+%d h, variants A / B / C ===\n", N_HORIZONS);

    if (!parse_input(WINDOW_JSON, input))
        return;
    print_input_summary(input.window);

    Forecast forecasts[N_VARIANTS] = {
        run_variant(Variant::A, input.window),
        run_variant(Variant::B, input.window),
        run_variant(Variant::C, input.window),
    };

    print_report(input, forecasts);
}
