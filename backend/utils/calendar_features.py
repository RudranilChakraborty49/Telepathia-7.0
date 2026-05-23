"""
calendar_features.py
--------------------
Calendar-based features capturing predictable F&O patterns.

Indian F&O expiry rules:
  - Until April 3, 2025:  Last Thursday of each month (monthly)
                          Every Thursday (weekly Nifty)
  - From April 4, 2025:   Last Tuesday of each month (monthly)
                          Every Tuesday (weekly Nifty)
"""

import pandas as pd
from datetime import datetime


# ───── EXPIRY DAY TRANSITION ─────
EXPIRY_DAY_SWITCH_DATE = datetime(2025, 4, 4)


def _expiry_weekday(date: datetime) -> int:
    """Returns Python weekday for F&O expiry: Tue=1 (new) or Thu=3 (old)."""
    return 1 if date >= EXPIRY_DAY_SWITCH_DATE else 3


def get_monthly_expiry(year: int, month: int) -> datetime:
    """Last expiry-day of a given month."""
    ref_date = datetime(year, month, 15)
    target_weekday = _expiry_weekday(ref_date)

    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - pd.Timedelta(days=1)

    offset = (last_day.weekday() - target_weekday) % 7
    return last_day - pd.Timedelta(days=offset)


def days_to_monthly_expiry(date: pd.Timestamp) -> int:
    """Days until next MONTHLY expiry."""
    this_month = get_monthly_expiry(date.year, date.month)
    if date.date() <= this_month.date():
        return (this_month.date() - date.date()).days

    if date.month == 12:
        next_expiry = get_monthly_expiry(date.year + 1, 1)
    else:
        next_expiry = get_monthly_expiry(date.year, date.month + 1)
    return (next_expiry.date() - date.date()).days


def days_to_weekly_expiry(date: pd.Timestamp) -> int:
    """Days until next WEEKLY expiry day."""
    target_weekday = _expiry_weekday(date.to_pydatetime())
    days_ahead = (target_weekday - date.weekday()) % 7
    return int(days_ahead)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all calendar features."""
    df = df.copy()

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Day_Of_Week"] = df["Date"].dt.dayofweek
    df["Days_To_Monthly_Expiry"] = df["Date"].apply(days_to_monthly_expiry)
    df["Days_To_Weekly_Expiry"] = df["Date"].apply(days_to_weekly_expiry)
    df["Is_Expiry_Day"] = (df["Days_To_Weekly_Expiry"] == 0).astype(int)
    df["Is_Monthly_Expiry_Week"] = (df["Days_To_Monthly_Expiry"] <= 7).astype(int)
    df["Month_Of_Year"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter

    return df


# ───── TEST ─────
if __name__ == "__main__":
    print("=" * 60)
    print("CALENDAR FEATURES — Thu→Tue expiry switch + weekly logic")
    print("=" * 60)

    print("\n--- Monthly expiry days ---")
    print(f"Jan 2024:  {get_monthly_expiry(2024, 1).strftime('%Y-%m-%d (%A)')}")
    print(f"Mar 2025:  {get_monthly_expiry(2025, 3).strftime('%Y-%m-%d (%A)')}")
    print(f"Apr 2025:  {get_monthly_expiry(2025, 4).strftime('%Y-%m-%d (%A)')}")
    print(f"May 2026:  {get_monthly_expiry(2026, 5).strftime('%Y-%m-%d (%A)')}")

    print("\n--- Days to weekly expiry ---")
    test_dates = [
        datetime(2024, 3, 25),
        datetime(2025, 5, 12),
        datetime(2025, 5, 13),
        datetime(2025, 5, 14),
    ]
    for d in test_dates:
        days = days_to_weekly_expiry(pd.Timestamp(d))
        print(f"  {d.strftime('%Y-%m-%d (%A)')}: {days} days to expiry")