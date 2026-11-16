#include "report.h"

#include <math.h>
#include <stdio.h>
#include <inttypes.h>

#include "esp_heap_caps.h"

namespace {

const char *const RULE =
    "  ------------------------------------------------------------------\n";

void print_cell(bool ok, float value)
{
    if (ok)
        printf("%8.3f", value);
    else
        printf("%8s", "-");
}

float mean_abs_error(const float *predicted, const float *measured)
{
    float sum = 0.0f;
    for (int h = 0; h < N_HORIZONS; h++)
        sum += fabsf(predicted[h] - measured[h]);
    return sum / N_HORIZONS;
}

void print_header(bool has_measured)
{
    printf("\n     time                 horizon        A        B        C");
    if (has_measured)
        printf("  measured");
    printf("\n%s", RULE);
}

void print_rows(const ForecastInput &in, const Forecast forecasts[N_VARIANTS])
{
    const window_t &w = in.window;
    for (int h = 0; h < N_HORIZONS; h++) {
        pv_datetime_t t = w.time[w.n - 1];
        cal_add_hours(&t, h + 1);

        printf("  %d  %04d-%02d-%02d %02d:%02d:00   +%d h  ",
               h, t.year, t.month, t.day, t.hour, t.minute, h + 1);
        for (int v = 0; v < N_VARIANTS; v++)
            print_cell(forecasts[v].ok, forecasts[v].kW[h]);
        if (in.has_measured)
            printf("%10.3f", in.measured[h]);
        printf("\n");
    }
    printf("%s", RULE);
}

void print_mae(const ForecastInput &in, const Forecast forecasts[N_VARIANTS])
{
    printf("  MAE [kW]                          ");
    for (int v = 0; v < N_VARIANTS; v++)
        print_cell(forecasts[v].ok, mean_abs_error(forecasts[v].kW, in.measured));
    printf("\n");
}

void print_latency(const Forecast forecasts[N_VARIANTS])
{
    printf("  latency us per run                ");
    for (int v = 0; v < N_VARIANTS; v++)
        printf("%8" PRId64, forecasts[v].us_per_run);
    printf("\n\n");
}

}  // namespace

void print_input_summary(const window_t &w)
{
    const pv_datetime_t &last = w.time[w.n - 1];
    printf("window: %d rows, ends %04d-%02d-%02d %02d:%02d UTC\n",
           w.n, last.year, last.month, last.day, last.hour, last.minute);
    printf("free heap: %u B\n\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));
}

void print_report(const ForecastInput &in, const Forecast forecasts[N_VARIANTS])
{
    print_header(in.has_measured);
    print_rows(in, forecasts);
    if (in.has_measured)
        print_mae(in, forecasts);
    print_latency(forecasts);
}
