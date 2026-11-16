#include "input_parser.h"

#include <stdio.h>
#include <memory>

#include "cJSON.h"

namespace {

// Channel keys in RAW_* order.
const char *const RAW_KEYS[N_RAW] = {
    "P_Solar[kW]", "GHI[kW1m2]", "POA Irr[kW1m2]", "TEMPERATURE[degC]",
    "HUMIDITY[%]", "WIND_SPEED[m1s]", "WIND_DIR[deg]",
};

using JsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

const cJSON *get_array(const cJSON *root, const char *key)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    return cJSON_IsArray(item) ? item : nullptr;
}

bool parse_timestamp(const char *text, pv_datetime_t &t)
{
    int seconds = 0;
    return sscanf(text, "%d-%d-%dT%d:%d:%d",
                  &t.year, &t.month, &t.day, &t.hour, &t.minute, &seconds) >= 5;
}

bool parse_timestamps(const cJSON *root, window_t &w)
{
    const cJSON *ts = get_array(root, "timestamps");
    if (!ts) {
        printf("no timestamps array\n");
        return false;
    }

    w.n = cJSON_GetArraySize(ts);
    if (w.n < MIN_ROWS || w.n > MAX_ROWS) {
        printf("need %d..%d rows, got %d\n", MIN_ROWS, MAX_ROWS, w.n);
        return false;
    }

    for (int i = 0; i < w.n; i++) {
        const cJSON *e = cJSON_GetArrayItem(ts, i);
        if (!cJSON_IsString(e) || !parse_timestamp(e->valuestring, w.time[i])) {
            printf("bad timestamp at index %d\n", i);
            return false;
        }
    }
    return true;
}

bool parse_channels(const cJSON *root, window_t &w)
{
    for (int ch = 0; ch < N_RAW; ch++) {
        const cJSON *arr = get_array(root, RAW_KEYS[ch]);
        if (!arr || cJSON_GetArraySize(arr) != w.n) {
            printf("channel '%s' missing or wrong length\n", RAW_KEYS[ch]);
            return false;
        }
        for (int i = 0; i < w.n; i++)
            w.raw[i][ch] = (float)cJSON_GetArrayItem(arr, i)->valuedouble;
    }
    return true;
}

// Optional -- absence is not an error.
bool parse_measured(const cJSON *root, float measured[N_HORIZONS])
{
    const cJSON *arr = get_array(root, "actual_next_6");
    if (!arr || cJSON_GetArraySize(arr) < N_HORIZONS)
        return false;

    for (int h = 0; h < N_HORIZONS; h++)
        measured[h] = (float)cJSON_GetArrayItem(arr, h)->valuedouble;
    return true;
}

}  // namespace

bool parse_input(const char *json, ForecastInput &in)
{
    JsonPtr root(cJSON_Parse(json), cJSON_Delete);
    if (!root) {
        printf("JSON parse error\n");
        return false;
    }

    if (!parse_timestamps(root.get(), in.window) ||
        !parse_channels(root.get(), in.window))
        return false;

    in.has_measured = parse_measured(root.get(), in.measured);
    return true;
}
