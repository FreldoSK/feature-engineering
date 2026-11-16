#include "calendar.h"

static const int DAYS_IN_MONTH[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
static const int DAYS_BEFORE_MONTH[12] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };

int cal_is_leap(int year)
{
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

static int days_in_month(int year, int month)
{
    return DAYS_IN_MONTH[month - 1] + (month == 2 && cal_is_leap(year));
}

int cal_day_of_year(const pv_datetime_t *t)
{
    int doy = DAYS_BEFORE_MONTH[t->month - 1] + t->day;
    if (t->month > 2 && cal_is_leap(t->year))
        doy += 1;
    return doy;
}

float cal_hour_of_day(const pv_datetime_t *t)
{
    return (float)t->hour + (float)t->minute / 60.0f;
}

void cal_add_hours(pv_datetime_t *t, int hours)
{
    t->hour += hours;
    while (t->hour >= 24) {
        t->hour -= 24;
        if (++t->day > days_in_month(t->year, t->month)) {
            t->day = 1;
            if (++t->month > 12) {
                t->month = 1;
                ++t->year;
            }
        }
    }
}
