// =============================================================================
// Minimal calendar arithmetic (Gregorian, no time zones).
// Pure C99, no ESP-IDF dependency -- can be unit-tested on a host.
// =============================================================================
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int year;
    int month;      // 1..12
    int day;        // 1..31
    int hour;       // 0..23
    int minute;     // 0..59
} pv_datetime_t;

int   cal_is_leap(int year);

// 1-based day of the year, matching pandas DatetimeIndex.dayofyear.
int   cal_day_of_year(const pv_datetime_t *t);

// Fractional hour of the day: hour + minute / 60.
float cal_hour_of_day(const pv_datetime_t *t);

// Adds `hours` (>= 0), rolling over day / month / year.
void  cal_add_hours(pv_datetime_t *t, int hours);

#ifdef __cplusplus
}
#endif
