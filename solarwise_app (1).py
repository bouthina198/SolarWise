"""
SolarWise - Climate-Based Smart Energy Usage Advisor
=====================================================
FTL Syria AI4Climate Hackathon, Team 7

One file:  PART 1 = data analysis with pandas   |   PART 2 = Tkinter interface

Run:
    pip install pandas numpy
    python solarwise_app.py

The dataset "damascus_nasa_hourly.csv" (NASA POWER, hourly) should sit next to
this file. If it is not found, the app asks you to locate it.

User inputs   : inverter (kW), battery (kWh), solar panels (kW), battery charge,
                devices to run + priority
Dataset inputs: ALLSKY_SFC_SW_DWN (main), CLRSKY_SFC_SW_DWN, T2M, PRECTOTCORR (optional)
Team data     : APPLIANCE_LIBRARY - average power of common household devices
"""
import math
import random
import datetime
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd


# =====================================================================
#  PART 1 - DATA ANALYSIS (pandas)
# =====================================================================
DATA_FILE = "damascus_nasa_hourly.csv"
DATA_LOCATION = "Damascus"
DATA_TIMEZONE = "Asia/Damascus"   # NASA POWER hourly timestamps are UTC

# NASA POWER column -> readable name
NASA_COLUMNS = {
    "ALLSKY_SFC_SW_DWN": "irradiance",   # W/m2 reaching the ground (MAIN input)
    "CLRSKY_SFC_SW_DWN": "clear_sky",    # W/m2 if the sky were clear (optional)
    "T2M": "temperature",                # air temperature at 2 m, degC (optional)
    "PRECTOTCORR": "rain",               # rainfall, mm/hour (optional)
}
NASA_MISSING = -999                       # NASA's code for "no value"

# Model assumptions (documented so the jury can see them)
PERFORMANCE_RATIO = 0.85    # wiring, dust, inverter losses
TEMP_COEFF = -0.004         # panels lose 0.4% power per degC above 25 degC
CELL_HEATING = 0.03         # cell temp = air temp + 0.03 x irradiance
BACKGROUND_KW = 0.15        # router, chargers, standby devices (always on)
EVENING_START = 18.0        # essential devices shorter than 24 h run in the evening
HIGH_CLEARNESS, LOW_CLEARNESS = 0.75, 0.55   # thresholds for HIGH / LOW solar days
CLOUDY_DROP = 0.25          # warn when tomorrow's solar is 25% lower than today
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d",
                "%d %b %Y", "%d %B %Y", "%d/%m/%y"]   # ways the user may type a date
CLIMATE_WINDOW = 7          # +/- days around a date used to build the typical day
CLOUD_CHANCE_ALERT = 0.12   # warn when 12% or more of past days around that date were cloudy
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

PRIORITIES = ["Essential", "Important", "Flexible"]

# Team-built table: average power of household devices (kW) and typical run time (h)
APPLIANCE_LIBRARY = [
    {"name": "Washing machine", "power": 1.5, "duration": 1.5, "priority": "Flexible"},
    {"name": "Dishwasher", "power": 1.8, "duration": 1.5, "priority": "Flexible"},
    {"name": "Electric water heater", "power": 2.0, "duration": 2.0, "priority": "Important"},
    {"name": "Water pump", "power": 0.75, "duration": 1.0, "priority": "Flexible"},
    {"name": "Air conditioner", "power": 2.0, "duration": 4.0, "priority": "Important"},
    {"name": "Electric oven", "power": 2.0, "duration": 1.0, "priority": "Important"},
    {"name": "Microwave", "power": 1.0, "duration": 0.25, "priority": "Flexible"},
    {"name": "Electric kettle", "power": 1.8, "duration": 0.25, "priority": "Flexible"},
    {"name": "Iron", "power": 1.2, "duration": 0.5, "priority": "Flexible"},
    {"name": "Vacuum cleaner", "power": 1.0, "duration": 0.5, "priority": "Flexible"},
    {"name": "TV", "power": 0.1, "duration": 4.0, "priority": "Flexible"},
    {"name": "Laptop charging", "power": 0.06, "duration": 3.0, "priority": "Flexible"},
    {"name": "Refrigerator", "power": 0.15, "duration": 24.0, "priority": "Essential"},
    {"name": "LED lighting", "power": 0.1, "duration": 6.0, "priority": "Essential"},
]


def fmt_time(t):
    h, m = int(t), int(round((t - int(t)) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h % 24:02d}:{m:02d}"


class SolarData:
    """Step 1-4 of the analysis: import, inspect, clean and prepare the dataset."""

    def __init__(self, path):
        self.path = Path(path)
        raw = pd.read_csv(self.path)                                   # 1. import
        self.raw_rows = len(raw)
        df = raw.rename(columns=NASA_COLUMNS)
        if "time" not in df.columns or "irradiance" not in df.columns:
            raise ValueError("The CSV needs a 'time' column and an ALLSKY_SFC_SW_DWN column.")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")        # 2. inspect / clean
        df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
        present = [c for c in NASA_COLUMNS.values() if c in df.columns]
        self.has = {c: c in present for c in NASA_COLUMNS.values()}
        df[present] = df[present].apply(pd.to_numeric, errors="coerce")
        df[present] = df[present].mask(df[present] <= NASA_MISSING)
        self.missing_values = int(df[present].isna().sum().sum())
        for c in NASA_COLUMNS.values():
            if c not in df.columns:
                df[c] = np.nan
        df = df.set_index("time")[list(NASA_COLUMNS.values())]

        self._utc = df.copy()                                           # kept for the 2026 outlook
        try:                                                            # 3. UTC -> local time
            df.index = df.index.tz_localize("UTC").tz_convert(DATA_TIMEZONE).tz_localize(None)
        except Exception:
            df.index = df.index + pd.Timedelta(hours=3)
        df = df[~df.index.duplicated()].asfreq("h")                    # fills clock-change gaps
        df = df.interpolate(limit=3, limit_direction="both")
        for c in ("irradiance", "clear_sky", "rain"):
            df[c] = df[c].clip(lower=0)
        df["irradiance"] = df["irradiance"].fillna(0)
        if not self.has["clear_sky"]:   # no clear-sky column: use the sunniest 5% for that month/hour
            df["clear_sky"] = df.groupby([df.index.month, df.index.hour])["irradiance"].transform(
                lambda s: s.quantile(.95))
        df["clear_sky"] = df["clear_sky"].fillna(df["irradiance"])
        self.hourly = df

        counts = df["irradiance"].resample("D").count()                 # 4. prepare daily table
        full_days = counts[counts == 24].index
        daily = df.resample("D").agg({"irradiance": "sum", "clear_sky": "sum",
                                      "temperature": "max", "rain": "sum"}).loc[full_days]
        daily = daily.rename(columns={"temperature": "t_max"})
        daily["sun_kwh_m2"] = daily["irradiance"] / 1000
        daily["clearness"] = (daily["irradiance"] / daily["clear_sky"].where(daily["clear_sky"] > 0)).clip(upper=1)
        self.daily = daily

        h = df[df.index.normalize().isin(full_days)]                    # day x hour tables
        h = h.assign(day=h.index.normalize(), hour=h.index.hour)
        self.irr_matrix = h.pivot(index="day", columns="hour", values="irradiance")
        self.temp_matrix = h.pivot(index="day", columns="hour", values="temperature")
        self.first_day = full_days.min().date()
        self.last_day = full_days.max().date()
        self.n_years = self.last_day.year - self.first_day.year + 1
        self._build_climatology()

    # ---- 2026 outlook: the "typical day" for each date, built from all past years
    @staticmethod
    def noleap_doy(idx):
        """Day of year 1..365 that ignores 29 Feb (29 Feb counts as 28 Feb)."""
        idx = pd.DatetimeIndex(idx)
        doy = np.asarray(idx.dayofyear)
        return doy - ((np.asarray(idx.is_leap_year)) & (doy >= 60)).astype(int)

    def _build_climatology(self):
        w = CLIMATE_WINDOW
        # Syria has used UTC+3 all year since 2022, so 2026 days are built on UTC+3
        h = self._utc.copy()
        h.index = h.index + pd.Timedelta(hours=3)
        h = h[h.index.normalize().isin(self.daily.index)].interpolate(limit=3)
        cols = ["irradiance", "clear_sky", "temperature", "rain"]
        # 1) average of every (day of year, hour) over all years
        mean = h.groupby([self.noleap_doy(h.index), h.index.hour])[cols].mean()
        self.clim = {}
        for c in cols:
            m = mean[c].unstack().reindex(index=range(1, 366), columns=range(24)).to_numpy()
            # 2) smooth with the days around it (+/- 7 days, wraps Dec -> Jan)
            stack = np.stack([np.roll(m, k, axis=0) for k in range(-w, w + 1)])
            self.clim[c] = np.nan_to_num(np.nanmean(stack, axis=0))
        # 3) daily statistics around each date: chance of cloud and a likely range
        d = self.daily
        dd = self.noleap_doy(d.index)
        sun, cloudy = d["sun_kwh_m2"].to_numpy(), (d["clearness"] < LOW_CLEARNESS).to_numpy()
        self.cloud_chance, self.sun_p10, self.sun_p90, self.sun_mean = (np.zeros(365) for _ in range(4))
        for k in range(1, 366):
            gap = np.abs(dd - k)
            sel = np.minimum(gap, 365 - gap) <= w
            self.cloud_chance[k - 1] = cloudy[sel].mean()
            self.sun_p10[k - 1], self.sun_p90[k - 1] = np.percentile(sun[sel], [10, 90])
            self.sun_mean[k - 1] = sun[sel].mean()
        del self._utc

    def is_expected(self, date):
        """True for any date the dataset does not cover (future or before 2015)."""
        return date > self.last_day or date < self.first_day

    def expected_stats(self, date):
        k = int(self.noleap_doy([pd.Timestamp(date)])[0]) - 1
        return {"cloud_chance": float(self.cloud_chance[k]),
                "low": float(self.sun_p10[k] / self.sun_mean[k]),
                "high": float(self.sun_p90[k] / self.sun_mean[k])}

    def day(self, date):
        """24 hourly rows (local time) for one date. Dates inside the dataset use the
        real data; any other date uses the typical day built from all years."""
        start = pd.Timestamp(date)
        hours = pd.date_range(start, periods=24, freq="h")
        if not self.is_expected(date):
            return self.hourly.reindex(hours)
        k = int(self.noleap_doy([start])[0]) - 1
        return pd.DataFrame({c: self.clim[c][k] for c in self.clim}, index=hours)

    def default_date(self):
        """Today's month/day in the most recent year of the dataset."""
        today = datetime.date.today()
        for year in range(self.last_day.year, self.first_day.year - 1, -1):
            try:
                d = datetime.date(year, today.month, min(today.day, 28 if today.month == 2 else today.day))
            except ValueError:
                continue
            if self.first_day <= d < self.last_day:
                return d
        return self.first_day

    def summary(self):
        fixed = f"{self.missing_values} missing values filled" if self.missing_values else "no missing values"
        return f"{self.raw_rows:,} hourly rows, {self.first_day.year} to {self.last_day.year}, {fixed}"


def parse_date(text):
    """Read a date typed by the user, e.g. 15/03/2026, 2026-03-15 or 15 Mar 2026.
    Without a year (15/03) the current year is used. Returns None if unreadable."""
    text = " ".join(text.strip().replace(",", " ").split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    for fmt in ("%d/%m", "%d-%m", "%d %b", "%d %B"):
        try:
            d = datetime.datetime.strptime(text, fmt)
            return datetime.date(datetime.date.today().year, d.month, d.day)
        except ValueError:
            pass
    return None


def pv_output(irradiance, temperature, system):
    """kW from the panels for each hour. Works on single days or whole tables."""
    g = np.asarray(irradiance, dtype=float)
    factor = 1.0
    if temperature is not None:
        t = np.asarray(temperature, dtype=float)
        if not np.all(np.isnan(t)):
            t_cell = np.nan_to_num(t, nan=25.0) + CELL_HEATING * g
            factor = 1 + TEMP_COEFF * (t_cell - 25)
    p = system["pv_kw"] * g / 1000 * PERFORMANCE_RATIO * factor
    return np.clip(p, 0, system["inverter_kw"])


def place_load(profile, power, start, duration):
    """Add a device running from `start` for `duration` hours to a 24-h profile."""
    for h in range(24):
        for shift in (0, 24):  # loads that pass midnight
            overlap = max(0.0, min(h + 1 + shift, start + duration) - max(h + shift, start))
            profile[h] += power * overlap
    return profile


def demand_profile(appliances):
    """Household demand (kW) per hour: background + essential devices."""
    prof = np.full(24, BACKGROUND_KW)
    for a in appliances:
        if a["priority"] == "Essential" and a.get("run"):
            if a["duration"] >= 24:
                prof += a["power"]
            else:
                place_load(prof, a["power"], EVENING_START, a["duration"])
    return prof


def window_stats(power, duration, start, gen, demand):
    samples = [start + i * .5 for i in range(max(1, round(duration * 2)))]
    g = np.array([gen[int(s) % 24] for s in samples])
    d = np.array([demand[int(s) % 24] for s in samples])
    covered = np.minimum(np.maximum(g - d, 0), power).sum() * .5
    return {"start": start, "end": start + duration, "solar": float(g.mean()),
            "demand": float(d.mean()), "surplus": float((g - d).mean() - power),
            "coverage": float(min(1.0, covered / (power * duration)))}


def best_window(power, duration, gen, demand, sunrise, sunset):
    """Slide the device across the daylight hours (30-min steps) and rank the slots."""
    first, last = math.floor(sunrise * 2) / 2, sunset - duration
    if last < first:
        first, last = 6.0, 18.0 - duration
    cands, t = [], first
    while t <= last + 1e-9:
        cands.append(window_stats(power, duration, t, gen, demand))
        t += .5
    if not cands:
        return None, None
    cands.sort(key=lambda c: (round(c["coverage"], 3), c["surplus"]), reverse=True)
    best = cands[0]
    alt = next((c for c in cands[1:] if c["start"] >= best["end"] or c["end"] <= best["start"]), None)
    return best, alt


def simulate_battery(gen, demand, capacity, soc_pct):
    energy, curve, full_at = capacity * soc_pct / 100, [], None
    for h in range(24):
        energy = min(capacity, max(0.0, energy + gen[h] - demand[h]))
        curve.append(round(energy / capacity * 100, 1))
        if full_at is None and gen[h] > 0 and energy >= capacity * .99:
            full_at = h + 1
    return curve, full_at


def sun_times(clear):
    """Sunrise / sunset estimated from the hourly clear-sky curve:
    solar noon = centre of the curve, day length = number of lit hours."""
    lit = [h for h in range(24) if clear[h] > 1]
    if not lit:
        return 6.0, 18.0
    total = sum(clear[h] for h in lit)
    noon = sum((h + .5) * clear[h] for h in lit) / total
    half = (lit[-1] + 1 - lit[0]) / 2
    return round((noon - half) * 12) / 12, round((noon + half) * 12) / 12


def analyze_day(data, system, appliances, date):
    """Everything the interface shows for one day."""
    day = data.day(date)
    irr = day["irradiance"].fillna(0).to_numpy()
    clear = day["clear_sky"].fillna(0).to_numpy()
    temp = day["temperature"].to_numpy()
    rain = day["rain"].fillna(0).to_numpy()
    gen = pv_output(irr, temp if data.has["temperature"] else None, system)
    demand = demand_profile(appliances)
    sunrise, sunset = sun_times(clear)

    windows, plan = {}, demand.copy()
    for a in appliances:
        if a["priority"] == "Essential" or a["duration"] >= 12:
            continue
        windows[a["name"]] = best_window(a["power"], a["duration"], gen, demand, sunrise, sunset)
        if a.get("run") and windows[a["name"]][0]:
            place_load(plan, a["power"], windows[a["name"]][0]["start"], a["duration"])

    scheduled = sum(a["power"] * a["duration"] for a in appliances
                    if a.get("run") and a["name"] in windows)
    soc_curve, full_at = simulate_battery(gen, plan, system["battery_kwh"], system["soc"])
    ratio = min(1.0, irr.sum() / clear.sum()) if clear.sum() > 0 else 0.0
    peak = int(np.argmax(gen))
    temp_loss = 0.0
    if data.has["temperature"] and irr[peak] > 0 and not np.isnan(temp[peak]):
        temp_loss = max(0.0, -TEMP_COEFF * (temp[peak] + CELL_HEATING * irr[peak] - 25))

    res = {
        "date": date,
        "weather": {"irradiance": irr.round(1).tolist(), "clear_sky": clear.round(1).tolist(),
                    "temperature": np.nan_to_num(temp).round(1).tolist(), "rain": rain.round(2).tolist()},
        "generation": gen.round(3).tolist(), "demand": demand.round(3).tolist(),
        "total_gen": float(gen.sum()), "total_demand": float(demand.sum() + scheduled),
        "scheduled_energy": float(scheduled),
        "balance": float(gen.sum() - demand.sum() - scheduled),
        "ratio": float(ratio),
        "level": "HIGH" if ratio >= HIGH_CLEARNESS else "MEDIUM" if ratio >= LOW_CLEARNESS else "LOW",
        "soc_curve": soc_curve, "battery_full_at": full_at, "peak_hour": peak,
        "sunrise": sunrise, "sunset": sunset,
        "max_temp": float(np.nanmax(temp)) if data.has["temperature"] else None,
        "rain_mm": float(rain.sum()), "temp_loss": temp_loss, "windows": windows,
        "mode": "actual",
    }
    if data.is_expected(date):
        st = data.expected_stats(date)
        res.update(mode="expected", cloud_chance=st["cloud_chance"],
                   gen_range=(res["total_gen"] * st["low"], res["total_gen"] * min(st["high"], 1.6)),
                   years=(data.first_day.year, data.last_day.year))
    res["recommendations"] = build_recommendations(appliances, res)
    return res


def build_recommendations(appliances, res):
    """Transparent if/else rules that turn the numbers into advice."""
    recs = []
    if res["mode"] == "expected":
        if res["cloud_chance"] >= CLOUD_CHANCE_ALERT:
            recs.append({"kind": "warn", "title": f"{res['cloud_chance']:.0%} chance of a cloudy day",
                         "text": "Keep some battery in reserve and don't plan every heavy load for today."})
    if res["level"] == "LOW":
        recs.append({"kind": "danger", "title": "Low solar generation expected",
                     "text": "Postpone flexible appliances if you can and avoid "
                             "running several heavy loads together."})
    if res["rain_mm"] >= 1:
        recs.append({"kind": "info", "title": f"Rain expected ({res['rain_mm']:.1f} mm)",
                     "text": "Output drops while it rains. Plan heavy loads for the dry hours."})
    for a in appliances:
        if not a.get("run") or a["name"] not in res["windows"]:
            continue
        best, _ = res["windows"][a["name"]]
        if not best:
            continue
        span = f"{fmt_time(best['start'])}-{fmt_time(best['end'])}"
        if best["coverage"] >= .99:
            recs.append({"kind": "success", "title": f"{a['name']}: {span}",
                         "text": f"About {best['solar']:.1f} kW of sun covers its {a['power']:g} kW load."})
        else:
            recs.append({"kind": "warn", "title": f"{a['name']}: {span}",
                         "text": f"Sunniest slot available. Solar covers about {best['coverage']:.0%}, "
                                 f"the rest comes from the battery."})
    if res["temp_loss"] >= .08:
        recs.append({"kind": "warn", "title": f"Hot day, panels lose about {res['temp_loss']:.0%}",
                     "text": "Heat lowers panel output at midday. Start heavy loads a little earlier."})
    if res["battery_full_at"] is not None:
        recs.append({"kind": "info", "title": f"Battery full by {res['battery_full_at']:02d}:00",
                     "text": "Solar after that is wasted unless you use it, so that is the moment for heavy loads."})
    else:
        recs.append({"kind": "warn", "title": "Battery won't reach 100%",
                     "text": "Keep evening use light to avoid a deep discharge overnight."})
    recs.append({"kind": "info", "title": f"No solar after {fmt_time(res['sunset'])}",
                 "text": "Avoid heavy appliances in the evening so the battery lasts the night."})
    if res["mode"] == "expected":
        y0, y1 = res["years"]
        recs.append({"kind": "info", "title": "Typical day, not a live forecast",
                     "text": f"Average of {y0}-{y1} around this date. Check the weather forecast on the day."})
    return recs


def tomorrow_warning(today, tomorrow):
    """'Cloudy tomorrow' alert when tomorrow's solar is 25% lower or more."""
    if tomorrow["mode"] == "expected":
        p = tomorrow["cloud_chance"]
        if p < CLOUD_CHANCE_ALERT:
            return None
        return {"kind": "warn", "title": f"{p:.0%} chance of a cloudy day tomorrow",
                "text": f"In past years, {p:.0%} of the days around this date were cloudy. "
                        f"Keep some battery in reserve and check the real forecast.", "drop": 0}
    if today["total_gen"] <= 0:
        return None
    drop = 1 - tomorrow["total_gen"] / today["total_gen"]
    if drop < CLOUDY_DROP:
        return None
    return {"kind": "danger", "title": "Cloudy day tomorrow",
            "text": f"Solar production is expected to be about {drop:.0%} lower than today "
                    f"({tomorrow['total_gen']:.1f} vs {today['total_gen']:.1f} kWh). Cut "
                    f"non-essential use and save battery energy overnight.", "drop": drop}


def historical_insights(data, system, appliances, habit_hour):
    """Analyses over the whole dataset (averages, min/max, thresholds, change, comparison)."""
    d = data.daily
    month = d.index.month
    monthly_sun = d.groupby(month)["sun_kwh_m2"].mean().reindex(range(1, 13))        # averages
    cloudy = (d["clearness"] < LOW_CLEARNESS)                                         # threshold
    monthly_cloudy = cloudy.groupby(month).mean().reindex(range(1, 13)) * 100

    gen_table = pv_output(data.irr_matrix.to_numpy(),
                          data.temp_matrix.to_numpy() if data.has["temperature"] else None, system)
    daily_kwh = pd.Series(gen_table.sum(axis=1), index=data.irr_matrix.index)
    monthly_kwh = daily_kwh.groupby(daily_kwh.index.month).mean().reindex(range(1, 13))

    days_per_year = d.groupby(d.index.year).size()                                   # change over time
    full_years = days_per_year[days_per_year >= 360].index
    yearly = d[d.index.year.isin(full_years)].groupby(d.index.year[d.index.year.isin(full_years)])["sun_kwh_m2"].sum()
    trend = (yearly.iloc[-1] / yearly.iloc[0] - 1) * 100 if len(yearly) >= 2 else 0.0

    rainy = d["rain"] >= 1                                                           # comparison
    rain_drop = (1 - d.loc[rainy, "clearness"].mean() / d.loc[~rainy, "clearness"].mean()) * 100 \
        if rainy.any() and (~rainy).any() else 0.0
    corr = d["t_max"].corr(d["sun_kwh_m2"]) if data.has["temperature"] else float("nan")

    res = {
        "years": (int(yearly.index[0]) if len(yearly) else data.first_day.year,
                  int(yearly.index[-1]) if len(yearly) else data.last_day.year),
        "avg_sun": float(d["sun_kwh_m2"].mean()), "avg_kwh": float(daily_kwh.mean()),
        "monthly_sun": monthly_sun.round(2).tolist(), "monthly_kwh": monthly_kwh.round(2).tolist(),
        "monthly_cloudy": monthly_cloudy.round(1).tolist(),
        "best_month": MONTHS[int(monthly_kwh.idxmax()) - 1], "worst_month": MONTHS[int(monthly_kwh.idxmin()) - 1],
        "best_kwh": float(monthly_kwh.max()), "worst_kwh": float(monthly_kwh.min()),
        "cloudy_days": float(cloudy.groupby(d.index.year).sum().loc[full_years].mean()) if len(full_years) else 0.0,
        "trend_pct": float(trend), "rain_drop_pct": float(rain_drop), "temp_sun_corr": float(corr),
        "compare": compare_scheduling(data, gen_table, system, appliances, habit_hour),
        "year_kwh": [float(v) for v in daily_kwh[daily_kwh.index.year.isin(full_years)]
                     .groupby(daily_kwh.index.year[daily_kwh.index.year.isin(full_years)]).sum()],
    }
    return res


def compare_scheduling(data, gen_table, system, appliances, habit_hour):
    """Normal (same hour every day) vs weather-based scheduling, over the latest full year."""
    devices = [a for a in appliances if a.get("run") and a["priority"] != "Essential" and a["duration"] < 12]
    if not devices:
        return None
    idx = data.irr_matrix.index
    year = idx.year.max() if (idx.year == idx.year.max()).sum() >= 365 else idx.year.max() - 1
    rows = np.where(idx.year == year)[0]
    surplus = np.maximum(gen_table[rows] - demand_profile(appliances), 0)   # days x 24

    def covered(power, duration, start):
        samples = [start + i * .5 for i in range(max(1, round(duration * 2)))]
        return sum(np.minimum(surplus[:, int(s) % 24], power) * .5 for s in samples)   # kWh per day

    solar_normal = solar_smart = total = 0.0
    for a in devices:
        p, dur = a["power"], a["duration"]
        starts = np.arange(5.0, 19.0 - dur + .01, .5)
        options = np.stack([covered(p, dur, s) for s in starts]) if len(starts) else covered(p, dur, 6.0)[None]
        solar_smart += options.max(axis=0).sum()
        solar_normal += covered(p, dur, habit_hour).sum()
        total += p * dur * len(rows)
    return {"year": int(year), "habit": habit_hour, "total_kwh": total,
            "normal_kwh": solar_normal, "smart_kwh": solar_smart,
            "normal_pct": solar_normal / total * 100, "smart_pct": solar_smart / total * 100}


def load_data(root=None):
    """Find the CSV next to this file (or in the working folder); otherwise ask for it."""
    here = Path(__file__).resolve().parent
    for p in (here / DATA_FILE, Path.cwd() / DATA_FILE):
        if p.exists():
            return SolarData(p)
    if root is not None:
        messagebox.showinfo("SolarWise", f"Please select the dataset file ({DATA_FILE}).", parent=root)
    path = filedialog.askopenfilename(title="Select the NASA POWER CSV",
                                      filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not path:
        raise SystemExit("No dataset selected.")
    return SolarData(path)


# =====================================================================
#  PART 2 - INTERFACE (Tkinter)
# =====================================================================
# ----------------------------------------------------------------- theme --
C = {
    "bg": "#0B1326", "side": "#09101F", "card": "#111C35", "card2": "#172447",
    "border": "#223360", "text": "#EDF1FA", "muted": "#8A96B5", "dim": "#56628A",
    "sun": "#FFB547", "sun_hover": "#FFC870", "ember": "#FF7F3F",
    "sky": "#6FD3FF", "green": "#3FE0A1", "red": "#FF6B81", "amber": "#FFA64D",
}
FONT = "Segoe UI"
KIND = {  # colour + symbol per message type
    "success": (C["green"], "\u2713"), "warn": (C["amber"], "!"),
    "danger": (C["red"], "\u2601"), "info": (C["sky"], "i"),
}
PRIORITY_COLORS = {"Essential": C["red"], "Important": C["sky"], "Flexible": C["green"]}
_FONTS = {}


def F(size, weight="normal"):
    return (FONT, size, weight)


def font_obj(size, weight="normal"):
    key = (size, weight)
    if key not in _FONTS:
        _FONTS[key] = tkfont.Font(family=FONT, size=size, weight=weight)
    return _FONTS[key]


def lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def soc_color(v):
    return C["green"] if v >= 60 else C["sun"] if v >= 30 else C["red"]


def round_rect(cv, x1, y1, x2, y2, r=16, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
           x2, y1 + r, x2, y2 - r, x2, y2 - r, x2, y2, x2 - r, y2, x2 - r, y2,
           x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r,
           x1, y1 + r, x1, y1]
    return cv.create_polygon(pts, smooth=True, **kw)


def mask_corners(cv, w, h, r, color):
    """Paint the outside of rounded corners so a full-bleed drawing looks rounded."""
    for cx, cy, a0, corner in ((r, r, 180, (0, 0)), (w - r, r, 270, (w, 0)),
                               (w - r, h - r, 0, (w, h)), (r, h - r, 90, (0, h))):
        pts = list(corner)
        for i in range(0, 91, 6):
            a = math.radians(a0 + i)
            pts += [cx + r * math.cos(a), cy + r * math.sin(a)]
        cv.create_polygon(pts, fill=color, outline=color)


def draw_sun(cv, cx, cy, r):
    for i in range(12):
        a = math.radians(i * 30)
        cv.create_line(cx + (r + 6) * math.cos(a), cy + (r + 6) * math.sin(a),
                       cx + (r + 13) * math.cos(a), cy + (r + 13) * math.sin(a),
                       fill="#FFD98A", width=3, capstyle="round")
    cv.create_oval(cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3, fill="#7A5A24", outline="")
    cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=C["sun"], outline="")
    cv.create_oval(cx - r * .55, cy - r * .6, cx + r * .25, cy + r * .2, fill="#FFD27F", outline="")


def draw_cloud(cv, cx, cy, s):
    for dy, col in ((4, "#9AA8C4"), (0, "#E6ECF7")):
        for x1, y1, x2, y2 in ((-1.1, -.2, -.1, .6), (-.6, -.75, .5, .45), (0, -.35, 1.0, .6)):
            cv.create_oval(cx + x1 * s, cy + y1 * s + dy, cx + x2 * s, cy + y2 * s + dy,
                           fill=col, outline="")
        cv.create_rectangle(cx - .7 * s, cy + .1 * s + dy, cx + .6 * s, cy + .6 * s + dy,
                            fill=col, outline="")


def pill(cv, x, y, text, fg, bg, size=10):
    tw = font_obj(size, "bold").measure(text)
    round_rect(cv, x, y - 15, x + tw + 26, y + 15, 15, fill=bg, outline="")
    cv.create_text(x + 13, y, text=text, anchor="w", fill=fg, font=F(size, "bold"))
    return x + tw + 26


# ------------------------------------------------------------ components --
class Card(tk.Canvas):
    """Rounded panel. Put widgets into `.inner`.
    fill=False -> height follows content;  fill=True -> content fills the card."""

    def __init__(self, master, pad=18, radius=16, color=None, border=None,
                 fill=False, height=None):
        super().__init__(master, bg=master.cget("bg"), highlightthickness=0, bd=0,
                         width=1, height=height or 2 * pad + 4)
        self.pad, self.radius, self.fill = pad, radius, fill
        self.color, self.border = color or C["card"], border or C["border"]
        self.inner = tk.Frame(self, bg=self.color)
        self._win = self.create_window(pad, pad, window=self.inner, anchor="nw")
        self.bind("<Configure>", self._redraw)
        if not fill:
            self.inner.bind("<Configure>", self._fit, add="+")
            self.after_idle(self._fit)

    def _fit(self, _e=None):
        h = self.inner.winfo_reqheight() + 2 * self.pad
        if int(float(self.cget("height"))) != h:
            self.configure(height=h)

    def _redraw(self, _e=None):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("bg")
        round_rect(self, 1, 1, w - 1, h - 1, self.radius, fill=self.color,
                   outline=self.border, tags="bg")
        self.tag_lower("bg")
        self.itemconfigure(self._win, width=max(1, w - 2 * self.pad))
        if self.fill:
            self.itemconfigure(self._win, height=max(1, h - 2 * self.pad))

    def set_border(self, color):
        self.border = color
        self._redraw()


class PillButton(tk.Canvas):
    STYLES = {
        "primary": (C["sun"], C["sun_hover"], C["bg"], ""),
        "ghost": (C["card2"], C["border"], C["text"], C["border"]),
        "danger": ("#3A1E33", "#4A2440", C["red"], "#6A2E48"),
    }

    def __init__(self, master, text, command=None, kind="primary", width=None, height=42):
        width = width or font_obj(10, "bold").measure(text) + 44
        super().__init__(master, width=width, height=height, bg=master.cget("bg"),
                         highlightthickness=0, cursor="hand2")
        self.text, self.command, self.kind = text, command, kind
        self.bind("<Enter>", lambda e: self._draw(True))
        self.bind("<Leave>", lambda e: self._draw(False))
        self.bind("<ButtonRelease-1>", lambda e: self.command and self.command())
        self._draw(False)

    def _draw(self, hover):
        fill, hov, fg, outline = self.STYLES[self.kind]
        w, h = int(self["width"]), int(self["height"])
        self.delete("all")
        round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=hov if hover else fill,
                   outline=outline or (hov if hover else fill))
        self.create_text(w / 2, h / 2, text=self.text, fill=fg, font=F(10, "bold"))


class Chip(tk.Canvas):
    def __init__(self, master, text, command=None, selected=False, color=None, height=34):
        w = font_obj(10, "bold").measure(text) + 32
        super().__init__(master, width=w, height=height, bg=master.cget("bg"),
                         highlightthickness=0, cursor="hand2")
        self.text, self.command, self.selected = text, command, selected
        self.color, self.hover = color or C["sun"], False
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))
        self.bind("<ButtonRelease-1>", lambda e: self.command and self.command())
        self.draw()

    def _hover(self, v):
        self.hover = v
        self.draw()

    def set_selected(self, v):
        self.selected = v
        self.draw()

    def draw(self):
        w, h = int(self["width"]), int(self["height"])
        self.delete("all")
        if self.selected:
            fill, outline, fg = self.color, self.color, C["bg"]
        else:
            fill = C["card2"]
            outline = self.color if self.hover else C["border"]
            fg = C["text"] if self.hover else C["muted"]
        round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=fill, outline=outline)
        self.create_text(w / 2, h / 2, text=self.text, fill=fg, font=F(10, "bold"))


class ChipGroup(tk.Frame):
    def __init__(self, master, options, value=None, command=None, per_row=6, colors=None):
        super().__init__(master, bg=master.cget("bg"))
        self.command, self.per_row, self.colors = command, per_row, colors or {}
        self.value, self.chips = value, {}
        self.set_options(options, value)

    def set_options(self, options, value=None):
        for ch in self.chips.values():
            ch.destroy()
        self.chips = {}
        if value is not None:
            self.value = value
        if options and self.value not in options:
            self.value = options[0]
        for i, o in enumerate(options):
            ch = Chip(self, o, command=lambda o=o: self.select(o),
                      selected=(o == self.value), color=self.colors.get(o))
            ch.grid(row=i // self.per_row, column=i % self.per_row,
                    padx=(0, 8), pady=(0, 8), sticky="w")
            self.chips[o] = ch

    def select(self, o, notify=True):
        self.value = o
        for k, ch in self.chips.items():
            ch.set_selected(k == o)
        if notify and self.command:
            self.command(o)


class Segmented(tk.Canvas):
    def __init__(self, master, options, command=None, width=220, height=40):
        super().__init__(master, width=width, height=height, bg=master.cget("bg"),
                         highlightthickness=0, cursor="hand2")
        self.options, self.command, self.value = options, command, options[0]
        self.bind("<ButtonRelease-1>", self._click)
        self.draw()

    def set(self, v):
        self.value = v
        self.draw()

    def draw(self):
        w, h = int(self["width"]), int(self["height"])
        self.delete("all")
        round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=C["card2"], outline=C["border"])
        seg = (w - 8) / len(self.options)
        for i, o in enumerate(self.options):
            x1 = 4 + i * seg
            on = o == self.value
            if on:
                round_rect(self, x1, 4, x1 + seg, h - 4, (h - 8) / 2, fill=C["sun"], outline="")
            self.create_text(x1 + seg / 2, h / 2, text=o, font=F(10, "bold"),
                             fill=C["bg"] if on else C["muted"])

    def _click(self, e):
        n = len(self.options)
        i = max(0, min(n - 1, int((e.x - 4) / ((int(self["width"]) - 8) / n))))
        if self.options[i] != self.value:
            self.set(self.options[i])
            if self.command:
                self.command(self.options[i])


class Slider(tk.Canvas):
    def __init__(self, master, value=50, command=None):
        super().__init__(master, height=34, width=1, bg=master.cget("bg"),
                         highlightthickness=0, cursor="hand2")
        self.value, self.command = value, command
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Button-1>", self._set)
        self.bind("<B1-Motion>", self._set)

    def _set(self, e):
        w = self.winfo_width()
        self.value = int(round(max(0, min(100, (e.x - 14) / max(1, w - 28) * 100))))
        self.draw()
        if self.command:
            self.command(self.value)

    def set(self, v):
        self.value = v
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 40:
            return
        p, y = 14, h / 2
        col = soc_color(self.value)
        round_rect(self, p, y - 4, w - p, y + 4, 4, fill=C["card2"], outline="")
        x = p + (w - 2 * p) * self.value / 100
        if x > p + 6:
            round_rect(self, p, y - 4, x, y + 4, 4, fill=col, outline="")
        self.create_oval(x - 11, y - 11, x + 11, y + 11, fill=C["text"], outline=col, width=4)


class Field(tk.Frame):
    def __init__(self, master, label, unit="", value=""):
        super().__init__(master, bg=master.cget("bg"))
        tk.Label(self, text=label, bg=self["bg"], fg=C["muted"], font=F(10)).pack(anchor="w", pady=(0, 6))
        self.box = Card(self, pad=10, radius=12, color=C["card2"])
        self.box.pack(fill="x")
        self.var = tk.StringVar(value=str(value))
        if unit:
            tk.Label(self.box.inner, text=unit, bg=C["card2"], fg=C["muted"],
                     font=F(10, "bold")).pack(side="right", padx=(0, 4))
        self.entry = tk.Entry(self.box.inner, textvariable=self.var, bg=C["card2"], fg=C["text"],
                              insertbackground=C["sun"], relief="flat", bd=0, width=6,
                              highlightthickness=0, font=F(13))
        self.entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.entry.bind("<FocusIn>", lambda e: self.box.set_border(C["sun"]))
        self.entry.bind("<FocusOut>", lambda e: self.box.set_border(C["border"]))

    def number(self, minimum=0.0):
        try:
            v = float(self.var.get().strip().replace(",", "."))
            if v <= minimum:
                raise ValueError
            return v
        except ValueError:
            self.box.set_border(C["red"])
            raise

    def text(self):
        return self.var.get().strip()


class Badge(tk.Canvas):
    def __init__(self, master, symbol, color, size=36):
        bg = master.cget("bg")
        super().__init__(master, width=size, height=size, bg=bg, highlightthickness=0)
        round_rect(self, 1, 1, size - 1, size - 1, size * .34, fill=lerp(color, bg, .8), outline="")
        self.create_text(size / 2, size / 2, text=symbol, fill=color, font=F(int(size * .36), "bold"))


class RingGauge(tk.Canvas):
    def __init__(self, master, size=74, thickness=8):
        super().__init__(master, width=size, height=size, bg=master.cget("bg"), highlightthickness=0)
        self.size, self.t = size, thickness

    def set(self, pct):
        s, t = self.size, self.t
        p = t / 2 + 2
        self.delete("all")
        self.create_oval(p, p, s - p, s - p, outline=C["card2"], width=t)
        ext = -3.6 * pct if pct < 100 else -359.9
        self.create_arc(p, p, s - p, s - p, start=90, extent=ext, style="arc",
                        outline=soc_color(pct), width=t)
        self.create_text(s / 2, s / 2, text=f"{pct:.0f}%", fill=C["text"], font=F(12, "bold"))


def auto_wrap(label, margin=4):
    """Let a label wrap to whatever width it gets."""
    label.bind("<Configure>", lambda e: label.config(wraplength=max(50, e.width - margin)), add="+")
    return label


def message_row(parent, msg, wrap=260):
    col, sym = KIND[msg["kind"]]
    bg = parent.cget("bg")
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x", pady=(0, 12))
    Badge(row, sym, col, size=32).pack(side="left", anchor="n")
    tx = tk.Frame(row, bg=bg)
    tx.pack(side="left", fill="x", expand=True, padx=(12, 0))
    tk.Label(tx, text=msg["title"], font=F(10, "bold"), fg=C["text"], bg=bg,
             anchor="w", justify="left", wraplength=wrap).pack(anchor="w")
    tk.Label(tx, text=msg["text"], font=F(9), fg=C["muted"], bg=bg, anchor="w",
             justify="left", wraplength=wrap).pack(anchor="w", pady=(2, 0))
    return row


# ------------------------------------------------------- drawn visuals --
class SunArc(tk.Canvas):
    """Dashboard hero: the sun's path across the sky, coloured by expected output."""

    def __init__(self, master, height=214):
        super().__init__(master, height=height, width=1, bg=master.cget("bg"),
                         highlightthickness=0, bd=0)
        self.data = None
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, **data):
        self.data = data
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 200 or not self.data:
            return
        d = self.data
        r, day, pv = d["res"], d["day"], d["pv"]
        lvl = r["level"]
        top = "#0E1834"
        glow = {"HIGH": "#5C3B1D", "MEDIUM": "#3F3448", "LOW": "#27344F"}[lvl]
        for y in range(0, h, 3):
            self.create_rectangle(0, y, w, y + 3, fill=lerp(top, glow, (y / h) ** 1.7), outline="")
        rnd = random.Random(3)
        for _ in range(28):
            x, y, s = rnd.uniform(w * .5, w), rnd.uniform(8, h * .5), rnd.choice((1, 1, 1.6))
            self.create_oval(x - s, y - s, x + s, y + s, outline="",
                             fill=lerp("#FFFFFF", top, rnd.uniform(.35, .8)))

        # text side
        head = {"HIGH": "Strong sun", "MEDIUM": "Patchy sun", "LOW": "Cloudy skies"}[lvl]
        if r.get("mode") == "expected":
            head = {"HIGH": "Sunny day expected", "MEDIUM": "Mixed day expected", "LOW": "Cloudy day likely"}[lvl]
        body = {
            "HIGH": f"Your {pv:g} kW panels should make about {r['total_gen']:.1f} kWh. "
                    f"Plenty for heavy appliances in the midday window.",
            "MEDIUM": f"Your {pv:g} kW panels should make about {r['total_gen']:.1f} kWh. "
                      f"Run heavy appliances only in the sunniest hours.",
            "LOW": f"Only about {r['total_gen']:.1f} kWh expected, {r['ratio']:.0%} of a "
                   f"clear day. Keep the battery for essentials.",
        }[lvl]
        if r.get("mode") == "expected":
            lo, hi = r["gen_range"]
            body = (f"Your {pv:g} kW panels should make about {r['total_gen']:.1f} kWh "
                    f"(usually {lo:.0f} to {hi:.0f}). Chance of a cloudy day: {r['cloud_chance']:.0%}.")
        self.create_text(32, 50, text=f"{head} {day}", anchor="w", fill=C["text"], font=F(24, "bold"))
        self.create_text(32, 80, text=body, anchor="nw", fill="#C9D2E8", font=F(11),
                         width=min(440, w * .44))
        lc = {"HIGH": C["sun"], "MEDIUM": C["amber"], "LOW": C["sky"]}[lvl]
        label = "Typical day" if r.get("mode") == "expected" else f"Solar availability: {lvl.title()}"
        x = pill(self, 32, h - 36, label, lc, lerp(lc, glow, .8))
        if d.get("best"):
            name, b = d["best"]
            pill(self, x + 10, h - 36,
                 f"{name}  {fmt_time(b['start'])}\u2013{fmt_time(b['end'])}",
                 C["green"], lerp(C["green"], glow, .82))

        # arc side
        x0, x1, horizon = w * .52, w - 56, h - 44
        cx, rx, ry = (x0 + x1) / 2, (x1 - x0) / 2, horizon - 40
        sr, ss = r["sunrise"], r["sunset"]

        def pt(t):
            a = math.pi * (1 - (t - sr) / (ss - sr))
            return cx + rx * math.cos(a), horizon - ry * math.sin(a)

        self.create_line(x0 - 26, horizon, x1 + 26, horizon, fill=lerp(C["sun"], glow, .55))
        dashed = []
        for i in range(61):
            dashed += pt(sr + (ss - sr) * i / 60)
        self.create_line(dashed, fill=C["dim"], dash=(2, 4))
        gen = r["generation"]
        gmax = max(max(gen), .01)
        n = 48
        for i in range(n):
            t0, t1 = sr + (ss - sr) * i / n, sr + (ss - sr) * (i + 1) / n
            f = gen[min(23, int((t0 + t1) / 2))] / gmax
            self.create_line(*pt(t0), *pt(t1), fill=lerp(C["dim"], C["sun"], f),
                             width=2 + 5 * f, capstyle="round")
        if d.get("best"):
            b = d["best"][1]
            s0, s1 = max(sr, b["start"]), min(ss, b["end"])
            seg = []
            for i in range(13):
                seg += pt(s0 + (s1 - s0) * i / 12)
            self.create_line(seg, fill=C["green"], width=9, capstyle="round", smooth=True)
        now = datetime.datetime.now()
        now_h = now.hour + now.minute / 60
        focus = now_h if (day == "today" and sr < now_h < ss) else r["peak_hour"] + .5
        sx, sy = pt(max(sr + .25, min(ss - .25, focus)))
        draw_sun(self, sx, sy, 15)
        if lvl != "HIGH":
            draw_cloud(self, sx + 12, sy + 10, 17 if lvl == "MEDIUM" else 25)
        self.create_text(pt(sr)[0], horizon + 18, text=f"Sunrise {fmt_time(sr)}",
                         fill=C["muted"], font=F(9))
        self.create_text(pt(ss)[0], horizon + 18, text=f"Sunset {fmt_time(ss)}",
                         fill=C["muted"], font=F(9))
        mask_corners(self, w, h, 22, self.master.cget("bg"))


class EnergyChart(tk.Canvas):
    """Hourly solar output (area) vs household demand (steps), with hover readout."""

    def __init__(self, master):
        super().__init__(master, width=1, height=1, bg=master.cget("bg"), highlightthickness=0)
        self.gen = self.dem = None
        self.window = None
        self.geom = None
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda e: self.delete("hover"))

    def set_data(self, gen, dem, window=None):
        self.gen, self.dem, self.window = gen, dem, window
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 80 or h < 60 or not self.gen:
            return
        L, R, T, B = 40, 24, 16, 26
        top_v = max(max(self.gen), max(self.dem), .5) * 1.2
        step = .5 if top_v <= 3 else 1.0
        vmax = math.ceil(top_v / step) * step
        X = lambda t: L + t / 24 * (w - L - R)
        Y = lambda v: h - B - v / vmax * (h - T - B)
        self.geom = (L, R, X, Y)
        bg = self["bg"]
        for i in range(int(round(vmax / step)) + 1):
            v = i * step
            self.create_line(L, Y(v), w - R, Y(v), fill=lerp(C["border"], bg, .35))
            self.create_text(L - 8, Y(v), text=f"{v:.1f}", anchor="e", fill=C["dim"], font=F(8))
        for hr in range(0, 25, 3):
            self.create_text(X(hr), h - B + 14, text=f"{hr:02d}:00", fill=C["dim"], font=F(8))
        if self.window:
            s, e = self.window["start"], self.window["end"]
            self.create_rectangle(X(s), T, X(e), h - B, fill=lerp(C["green"], bg, .86), outline="")
            self.create_line(X(s), T, X(s), h - B, fill=lerp(C["green"], bg, .5))
            self.create_line(X(e), T, X(e), h - B, fill=lerp(C["green"], bg, .5))
        pts = []
        for i, g in enumerate(self.gen):
            pts += [X(i + .5), Y(g)]
        base = Y(0)
        area = [X(.5), base, X(.5), base] + pts + [X(23.5), base, X(23.5), base]
        self.create_polygon(area, fill=lerp(C["sun"], bg, .8), outline="", smooth=True)
        self.create_line(pts, fill=C["sun"], width=3, smooth=True)
        steps = []
        for i, dv in enumerate(self.dem):
            steps += [X(i), Y(dv), X(i + 1), Y(dv)]
        self.create_line(steps, fill=C["sky"], width=2, dash=(5, 3))
        pk = max(range(24), key=lambda i: self.gen[i])
        px, py = X(pk + .5), Y(self.gen[pk])
        self.create_oval(px - 5, py - 5, px + 5, py + 5, fill=C["sun"], outline=bg, width=2)
        self.create_text(px, py - 14, text=f"{self.gen[pk]:.1f} kW", fill=C["sun"], font=F(9, "bold"))

    def _hover(self, e):
        self.delete("hover")
        if not self.geom:
            return
        L, R, X, Y = self.geom
        w, h = self.winfo_width(), self.winfo_height()
        hr = int((e.x - L) / max(1, w - L - R) * 24)
        if not 0 <= hr < 24:
            return
        x = X(hr + .5)
        self.create_line(x, 16, x, h - 26, fill=C["muted"], dash=(2, 3), tags="hover")
        for v, col in ((self.gen[hr], C["sun"]), (self.dem[hr], C["sky"])):
            self.create_oval(x - 4, Y(v) - 4, x + 4, Y(v) + 4, fill=col, outline="", tags="hover")
        bx = x + 14 if x < w - 190 else x - 174
        round_rect(self, bx, 18, bx + 160, 94, 10, fill=C["card2"], outline=C["border"], tags="hover")
        self.create_text(bx + 12, 34, text=f"{hr:02d}:00 to {hr + 1:02d}:00", anchor="w",
                         fill=C["text"], font=F(9, "bold"), tags="hover")
        self.create_text(bx + 12, 56, text=f"Solar   {self.gen[hr]:.2f} kW", anchor="w",
                         fill=C["sun"], font=F(9), tags="hover")
        self.create_text(bx + 12, 76, text=f"Home    {self.dem[hr]:.2f} kW", anchor="w",
                         fill=C["sky"], font=F(9), tags="hover")


class Timeline(tk.Canvas):
    """Best-time page: one block per daylight hour, brighter = more sun."""

    def __init__(self, master):
        super().__init__(master, height=112, width=1, bg=master.cget("bg"), highlightthickness=0)
        self.data = None
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, gen, best, alt, show_now):
        self.data = (gen, best, alt, show_now)
        self.draw()

    def draw(self):
        self.delete("all")
        w = self.winfo_width()
        if w < 100 or not self.data:
            return
        gen, best, alt, show_now = self.data
        s0, s1, top, bh = 5, 20, 30, 46
        X = lambda t: 6 + (t - s0) / (s1 - s0) * (w - 12)
        gmax = max(max(gen), .01)
        for hr in range(s0, s1):
            f = gen[hr] / gmax
            col = lerp(C["card2"], C["sun"], f ** .8) if gen[hr] > .02 else C["card2"]
            round_rect(self, X(hr) + 2, top, X(hr + 1) - 2, top + bh, 8, fill=col, outline="")
            self.create_text((X(hr) + X(hr + 1)) / 2, top + bh + 16, text=f"{hr:02d}",
                             fill=C["dim"], font=F(8))
        for win, col, label, width in ((alt, C["sky"], "Alternative", 2), (best, C["green"], "Best", 3)):
            if not win:
                continue
            a, b = X(max(s0, win["start"])), X(min(s1, win["end"]))
            round_rect(self, a - 1, top - 5, b + 1, top + bh + 5, 10, fill="", outline=col, width=width)
            self.create_text((a + b) / 2, top - 16, text=label, fill=col, font=F(9, "bold"))
        if show_now:
            n = datetime.datetime.now()
            t = n.hour + n.minute / 60
            if s0 <= t <= s1:
                self.create_line(X(t), top - 6, X(t), top + bh + 6, fill=C["red"], width=2)


class FlowDiagram(tk.Canvas):
    """My-system page: animated energy flow sun -> panels -> inverter -> home / battery."""

    def __init__(self, master, app):
        super().__init__(master, width=1, height=1, bg=master.cget("bg"), highlightthickness=0)
        self.app, self.phase, self.paths = app, 0.0, []
        self.bind("<Configure>", lambda e: self.draw())
        self.after(60, self._animate)

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 200 or h < 200:
            return
        if not self.app.ready:
            self.paths = []
            self.create_text(w / 2, h / 2, text="Save your system details to see how\nenergy moves through your home.",
                             fill=C["dim"], font=F(11), justify="center")
            return
        s, r = self.app.system, self.app.res["today"]
        pos = {"sun": (.17, .2), "pv": (.17, .55), "inv": (.53, .55), "home": (.8, .24), "bat": (.8, .86)}
        P = {k: (x * w, y * h) for k, (x, y) in pos.items()}
        self.paths = []
        for a, b, col in (("sun", "pv", C["sun"]), ("pv", "inv", C["sun"]),
                          ("inv", "home", C["sky"]), ("inv", "bat", C["green"])):
            (x1, y1), (x2, y2) = P[a], P[b]
            self.create_line(x1, y1, x2, y2, fill=lerp(col, self["bg"], .7), width=3)
            self.paths.append((x1, y1, x2, y2, col))
        nodes = {
            "sun": ("Sun", f"{max(r['weather']['irradiance']):.0f} W/m\u00b2 peak", C["sun"]),
            "pv": ("Solar panels", f"{s['pv_kw']:g} kW", C["sun"]),
            "inv": ("Inverter", f"{s['inverter_kw']:g} kW max", C["ember"]),
            "home": ("Home", f"{r['total_demand']:.1f} kWh/day", C["sky"]),
            "bat": ("Battery", f"{s['soc']}% of {s['battery_kwh']:g} kWh", soc_color(s["soc"])),
        }
        bw, bh = min(184, w * .31), 74
        for k, (title, value, col) in nodes.items():
            x, y = P[k]
            round_rect(self, x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, 14,
                       fill=C["card2"], outline=lerp(col, C["card2"], .45), width=2, tags="node")
            ix, iy = x - bw / 2 + 26, y
            if k == "sun":
                self.create_oval(ix - 11, iy - 11, ix + 11, iy + 11, fill=col, outline="", tags="node")
            elif k == "pv":
                for gx in range(3):
                    for gy in range(2):
                        self.create_rectangle(ix - 13 + gx * 9, iy - 9 + gy * 9, ix - 6 + gx * 9,
                                              iy - 2 + gy * 9, fill=col, outline="", tags="node")
            elif k == "inv":
                self.create_text(ix, iy, text="\u26a1", fill=col, font=F(18, "bold"), tags="node")
            elif k == "home":
                self.create_polygon(ix - 12, iy - 1, ix, iy - 12, ix + 12, iy - 1, fill=col, tags="node")
                self.create_rectangle(ix - 8, iy - 1, ix + 8, iy + 11, fill=col, outline="", tags="node")
            else:
                self.create_rectangle(ix - 13, iy - 8, ix + 11, iy + 8, outline=col, width=2, tags="node")
                self.create_rectangle(ix + 11, iy - 3, ix + 14, iy + 3, fill=col, outline="", tags="node")
                self.create_rectangle(ix - 11, iy - 6, ix - 11 + 20 * s["soc"] / 100, iy + 6,
                                      fill=col, outline="", tags="node")
            self.create_text(ix + 22, y - 11, text=title, anchor="w", fill=C["muted"],
                             font=F(9), tags="node")
            self.create_text(ix + 22, y + 10, text=value, anchor="w", fill=C["text"],
                             font=F(10, "bold"), tags="node")

    def _animate(self):
        self.phase = (self.phase + .01) % 1
        self.delete("dot")
        if self.winfo_ismapped():
            for x1, y1, x2, y2, col in self.paths:
                for k in range(3):
                    t = (self.phase + k / 3) % 1
                    x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                    self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=col, outline="", tags="dot")
            self.tag_raise("node")
        self.after(40, self._animate)


class DateNav(tk.Frame):
    """Type a date (then Enter) or step one day with the arrows."""

    def __init__(self, master, app, cloudy=True):
        super().__init__(master, bg=master.cget("bg"))
        self.app = app
        PillButton(self, "\u2039", lambda: app.shift_date(-1), kind="ghost", width=36, height=40).pack(side="left")
        self.box = Card(self, pad=8, radius=20, color=C["card2"])
        self.box.pack(side="left", padx=4)
        self.var = tk.StringVar()
        self.entry = tk.Entry(self.box.inner, textvariable=self.var, width=11, justify="center",
                              bg=C["card2"], fg=C["text"], insertbackground=C["sun"], relief="flat",
                              bd=0, highlightthickness=0, font=F(11, "bold"))
        self.entry.pack(padx=6)
        self.box.configure(width=130)
        self.entry.bind("<Return>", self.apply)
        self.entry.bind("<KP_Enter>", self.apply)
        self.entry.bind("<FocusIn>", lambda e: (self.box.set_border(C["sun"]), self.entry.select_range(0, "end")))
        self.entry.bind("<FocusOut>", lambda e: (self.box.set_border(C["border"]), self.refresh()))
        PillButton(self, "\u203a", lambda: app.shift_date(1), kind="ghost", width=36, height=40).pack(side="left")
        if cloudy:
            PillButton(self, "\u2601 Cloudy day", app.jump_cloudy, kind="ghost", height=40).pack(
                side="left", padx=(8, 0))

    def apply(self, _e=None):
        d = parse_date(self.var.get())
        if d is None:
            self.box.set_border(C["red"])
            self.app.toast("Date not recognised", "Type it like 15/03/2026, then press Enter.", "danger")
            return
        self.app.set_date(d)
        self.app.focus_set()

    def refresh(self):
        if self.focus_get() is not self.entry:
            self.var.set(self.app.shown_date().strftime("%d/%m/%Y"))


class BarChart(tk.Canvas):
    """Monthly bars with value labels; best month in amber, weakest in blue."""

    def __init__(self, master):
        super().__init__(master, width=1, height=1, bg=master.cget("bg"), highlightthickness=0)
        self.data = None
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, labels, values, extra=None):
        self.data = (labels, values, extra)
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if not self.data or w < 120 or h < 80:
            return
        labels, values, extra = self.data
        L, R, T, B = 6, 6, 24, 46
        vmax = max(values) * 1.12 or 1
        slot = (w - L - R) / len(values)
        bw = slot * .58
        imax, imin = values.index(max(values)), values.index(min(values))
        for i, v in enumerate(values):
            x = L + slot * i + slot / 2
            y = h - B - v / vmax * (h - T - B)
            col = C["sun"] if i == imax else C["sky"] if i == imin else lerp(C["card2"], C["sun"], .3 + .45 * v / max(values))
            round_rect(self, x - bw / 2, y, x + bw / 2, h - B, min(8, bw / 2), fill=col, outline="")
            self.create_rectangle(x - bw / 2, h - B - 6, x + bw / 2, h - B, fill=col, outline="")
            self.create_text(x, y - 10, text=f"{v:.1f}", fill=C["muted"], font=F(8, "bold"))
            self.create_text(x, h - B + 14, text=labels[i], fill=C["text"], font=F(9))
            if extra and extra[i] >= 1:
                self.create_text(x, h - B + 32, text=f"\u2601{extra[i]:.0f}%", fill=C["dim"], font=F(8))


class CompareBars(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, height=150, width=1, bg=master.cget("bg"), highlightthickness=0)
        self.rows = []
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, rows):
        self.rows = rows
        self.draw()

    def draw(self):
        self.delete("all")
        w = self.winfo_width()
        if w < 100:
            return
        for i, (label, pct, col) in enumerate(self.rows):
            y = 14 + i * 66
            self.create_text(0, y, text=label, anchor="w", fill=C["muted"], font=F(10))
            self.create_text(w, y, text=f"{pct:.0f}%", anchor="e", fill=col, font=F(14, "bold"))
            round_rect(self, 0, y + 18, w, y + 40, 11, fill=C["card2"], outline="")
            if pct > 2:
                round_rect(self, 0, y + 18, max(22, w * min(pct, 100) / 100), y + 40, 11, fill=col, outline="")


class Toast(tk.Toplevel):
    """In-app notification that slides in at the bottom-right of the window."""

    def __init__(self, app, title, text, kind="info"):
        super().__init__(app)
        self.app = app
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        key = "#010203"
        try:
            self.attributes("-transparentcolor", key)
            bg = key
        except tk.TclError:
            bg = C["bg"]
        self.configure(bg=bg)
        col, sym = KIND[kind]
        W = 380
        cv = tk.Canvas(self, width=W, height=100, bg=bg, highlightthickness=0)
        cv.pack()
        cv.create_text(80, 28, text=title, anchor="w", fill=C["text"], font=F(11, "bold"))
        body = cv.create_text(80, 42, text=text, anchor="nw", fill=C["muted"], font=F(9), width=W - 110)
        H = max(92, cv.bbox(body)[3] + 18)
        cv.configure(height=H)
        cv.create_text(W - 18, 18, text="\u2715", fill=C["dim"], font=F(10))
        cv.create_text(40, H / 2, text=sym, fill=col, font=F(18, "bold"), tags="badge")
        round_rect(cv, 2, 2, W - 2, H - 2, 16, fill=C["card2"], outline=lerp(col, C["card2"], .4),
                   width=2, tags="back")
        round_rect(cv, 16, H / 2 - 24, 64, H / 2 + 24, 14, fill=lerp(col, C["card2"], .78), outline="", tags="back")
        cv.tag_lower("back")
        cv.bind("<Button-1>", lambda e: self.close())
        self.W, self.H, self.alpha = W, H, 0.0
        self.place_me()
        self._fade(1)
        self.after(5200, self.close)

    def place_me(self):
        i = self.app.toasts.index(self) if self in self.app.toasts else len(self.app.toasts)
        x = self.app.winfo_rootx() + self.app.winfo_width() - self.W - 24
        y = self.app.winfo_rooty() + self.app.winfo_height() - (self.H + 12) * (i + 1) - 12
        self.geometry(f"+{x}+{y}")

    def _fade(self, direction):
        self.alpha = max(0.0, min(1.0, self.alpha + .12 * direction))
        try:
            self.attributes("-alpha", self.alpha)
        except tk.TclError:
            pass
        if 0 < self.alpha < 1:
            self.after(16, lambda: self._fade(direction))
        elif self.alpha <= 0:
            self._destroy()

    def close(self):
        if self.winfo_exists():
            self._fade(-1) if self.alpha > 0 else self._destroy()

    def _destroy(self):
        if self in self.app.toasts:
            self.app.toasts.remove(self)
        if self.winfo_exists():
            self.destroy()
        for t in self.app.toasts:
            t.place_me()


class NavItem(tk.Frame):
    def __init__(self, master, icon, text, command):
        super().__init__(master, bg=C["side"], cursor="hand2")
        self.active = False
        self.bar = tk.Frame(self, bg=C["side"], width=4)
        self.bar.pack(side="left", fill="y")
        self.icon = tk.Label(self, text=icon, font=F(13), width=2, bg=C["side"], fg=C["muted"])
        self.icon.pack(side="left", padx=(14, 8), pady=10)
        self.lbl = tk.Label(self, text=text, font=F(11, "bold"), bg=C["side"], fg=C["muted"])
        self.lbl.pack(side="left")
        self.badge = tk.Label(self, text="", font=F(8, "bold"), bg=C["red"], fg=C["bg"], padx=6)
        for wdg in (self, self.icon, self.lbl, self.badge):
            wdg.bind("<Button-1>", lambda e: command())
            wdg.bind("<Enter>", lambda e: self._paint(hover=True))
            wdg.bind("<Leave>", lambda e: self._paint())

    def set_active(self, v):
        self.active = v
        self._paint()

    def set_badge(self, n):
        if n:
            self.badge.config(text=str(n))
            self.badge.pack(side="right", padx=12)
        else:
            self.badge.pack_forget()

    def _paint(self, hover=False):
        bg = C["card"] if self.active else ("#0F1830" if hover else C["side"])
        fg = C["text"] if (self.active or hover) else C["muted"]
        for wdg in (self, self.icon, self.lbl):
            wdg.config(bg=bg)
        self.icon.config(fg=C["sun"] if self.active else fg)
        self.lbl.config(fg=fg)
        self.bar.config(bg=C["sun"] if self.active else bg)


# ----------------------------------------------------------------- pages --
class Page(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self.build()

    def header(self, title, subtitle):
        h = tk.Frame(self, bg=C["bg"])
        h.pack(fill="x", pady=(0, 18))
        left = tk.Frame(h, bg=C["bg"])
        left.pack(side="left")
        tk.Label(left, text=title, font=F(21, "bold"), fg=C["text"], bg=C["bg"]).pack(anchor="w")
        self.sub_lbl = tk.Label(left, text=subtitle, font=F(10), fg=C["muted"], bg=C["bg"])
        self.sub_lbl.pack(anchor="w", pady=(2, 0))
        self.header_right = tk.Frame(h, bg=C["bg"])
        self.header_right.pack(side="right")

    def body(self):
        b = tk.Frame(self, bg=C["bg"])
        b.pack(fill="both", expand=True)
        return b

    def build(self):
        pass

    def refresh(self):
        pass


class Metric(tk.Frame):
    def __init__(self, master, label, gauge=False):
        super().__init__(master, bg=master.cget("bg"))
        bg = self["bg"]
        if gauge:
            self.gauge = RingGauge(self)
            self.gauge.pack(side="left", padx=(0, 14))
        box = tk.Frame(self, bg=bg)
        box.pack(side="left", fill="x", expand=True)
        tk.Label(box, text=label, font=F(10), fg=C["muted"], bg=bg).pack(anchor="w")
        row = tk.Frame(box, bg=bg)
        row.pack(anchor="w", pady=(4, 2))
        self.val = tk.Label(row, font=F(22, "bold"), fg=C["text"], bg=bg)
        self.val.pack(side="left")
        self.unit = tk.Label(row, font=F(10, "bold"), fg=C["muted"], bg=bg)
        self.unit.pack(side="left", anchor="s", pady=(0, 6), padx=(4, 0))
        self.note = tk.Label(box, font=F(9), fg=C["muted"], bg=bg)
        self.note.pack(anchor="w")

    def set(self, value, unit="", note="", color=None):
        self.val.config(text=value)
        self.unit.config(text=unit)
        self.note.config(text=note, fg=color or C["muted"])


class Dashboard(Page):
    def build(self):
        self.header("Energy dashboard", "NASA POWER weather for this day, applied to your system")
        self.seg = Segmented(self.header_right, ["Today", "Tomorrow"], width=200,
                             command=lambda v: self.app.set_day(v.lower()))
        self.seg.pack(side="right")
        self.datenav = DateNav(self.header_right, self.app)
        self.datenav.pack(side="right", padx=16)

        b = self.body()
        for c in range(3):
            b.columnconfigure(c, weight=1, uniform="d")
        b.rowconfigure(2, weight=1)
        self.arc = SunArc(b)
        self.arc.grid(row=0, column=0, columnspan=3, sticky="ew")

        strip = Card(b, pad=20)
        strip.grid(row=1, column=0, columnspan=3, sticky="ew", pady=16)
        inn = strip.inner
        self.m = {}
        for i, (key, label) in enumerate((("gen", "Solar production"), ("use", "Home consumption"),
                                          ("bal", "Energy balance"), ("bat", "Battery at start"))):
            inn.columnconfigure(i * 2, weight=1)
            if i:
                tk.Frame(inn, bg=C["border"], width=1).grid(row=0, column=i * 2 - 1, sticky="ns", padx=18)
            self.m[key] = Metric(inn, label, gauge=(key == "bat"))
            self.m[key].grid(row=0, column=i * 2, sticky="w")

        chart_card = Card(b, fill=True, height=320)
        chart_card.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=(0, 8))
        top = tk.Frame(chart_card.inner, bg=C["card"])
        top.pack(fill="x", pady=(0, 8))
        tk.Label(top, text="Solar output vs home demand", font=F(13, "bold"), fg=C["text"],
                 bg=C["card"]).pack(side="left")
        for txt, col in (("\u25a0 Best window", C["green"]), ("\u2505 Home demand", C["sky"]),
                         ("\u2501 Solar", C["sun"])):
            tk.Label(top, text=txt, font=F(9, "bold"), fg=col, bg=C["card"]).pack(side="right", padx=(12, 0))
        self.chart = EnergyChart(chart_card.inner)
        self.chart.pack(fill="both", expand=True)

        self.rec_card = Card(b, fill=True, height=320)
        self.rec_card.grid(row=2, column=2, sticky="nsew", padx=(8, 0))
        tk.Label(self.rec_card.inner, text="What to do", font=F(13, "bold"), fg=C["text"],
                 bg=C["card"]).pack(anchor="w", pady=(0, 12))
        self.rec_box = tk.Frame(self.rec_card.inner, bg=C["card"])
        self.rec_box.pack(fill="both", expand=True)

    def refresh(self):
        app = self.app
        if not app.ready:
            return
        r, s = app.res[app.day], app.system
        self.seg.set(app.day.title())
        self.datenav.refresh()
        self.sub_lbl.config(text=f"Typical day from {app.data.first_day.year}\u2013{app.data.last_day.year} NASA data (no data for this date)"
                            if r["mode"] == "expected" else
                            "NASA POWER weather for this day, applied to your system")
        focus = app.focus_window(app.day)
        self.arc.set(res=r, day=app.day, pv=s["pv_kw"], best=focus)

        gen = r["generation"]
        self.m["gen"].set(f"{r['total_gen']:.1f}", "kWh",
                          f"Peak {max(gen):.1f} kW around {r['peak_hour']:02d}:00", C["sun"])
        self.m["use"].set(f"{r['total_demand']:.1f}", "kWh",
                          f"Includes {r['scheduled_energy']:.1f} kWh of planned appliances")
        bal = r["balance"]
        self.m["bal"].set(f"{bal:+.1f}", "kWh",
                          "Surplus: run flexible loads on sun" if bal >= 0
                          else "Deficit: battery will cover the gap",
                          C["green"] if bal >= 0 else C["red"])
        full = r["battery_full_at"]
        self.m["bat"].gauge.set(s["soc"])
        self.m["bat"].set(f"{s['battery_kwh'] * s['soc'] / 100:.1f}", "kWh",
                          f"Full by {full:02d}:00" if full else "Won't fill up today",
                          C["green"] if full else C["amber"])
        self.chart.set_data(gen, r["demand"], focus[1] if focus else None)

        for wdg in self.rec_box.winfo_children():
            wdg.destroy()
        recs = [m for m in r["recommendations"]
                if not (app.warning and app.day == "tomorrow" and "chance of a cloudy day" in m["title"])]
        msgs = ([app.warning] if app.warning else []) + recs
        for m in msgs[:3]:
            message_row(self.rec_box, m, wrap=250)
        if len(msgs) > 3:
            link = tk.Label(self.rec_box, text=f"See all {len(msgs)} tips in Notifications", cursor="hand2",
                            font=F(9, "bold"), fg=C["sun"], bg=C["card"])
            link.pack(anchor="w", pady=(2, 0))
            link.bind("<Button-1>", lambda e: app.show("alerts"))


class SystemPage(Page):
    def build(self):
        self.header("My solar system", "Your panels, inverter and battery. Every recommendation uses these")
        b = self.body()
        b.columnconfigure(0, weight=5, uniform="s")
        b.columnconfigure(1, weight=6, uniform="s")
        b.rowconfigure(0, weight=1)

        form = Card(b, fill=True, pad=24)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        f = form.inner
        tk.Label(f, text="System details", font=F(14, "bold"), fg=C["text"], bg=C["card"]).pack(anchor="w")
        tk.Label(f, text="Use the numbers printed on your equipment.", font=F(10),
                 fg=C["muted"], bg=C["card"]).pack(anchor="w", pady=(2, 18))
        grid = tk.Frame(f, bg=C["card"])
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1, uniform="g")
        grid.columnconfigure(1, weight=1, uniform="g")
        self.f_pv = Field(grid, "Solar panels (total)", "kW")
        self.f_inv = Field(grid, "Inverter rating", "kW")
        self.f_bat = Field(grid, "Battery capacity", "kWh")
        self.f_pv.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 14))
        self.f_inv.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=(0, 14))
        self.f_bat.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(0, 14))

        socrow = tk.Frame(f, bg=C["card"])
        socrow.pack(fill="x", pady=(6, 0))
        tk.Label(socrow, text="Battery charge right now", font=F(10), fg=C["muted"],
                 bg=C["card"]).pack(side="left")
        self.soc_lbl = tk.Label(socrow, font=F(11, "bold"), fg=C["text"], bg=C["card"])
        self.soc_lbl.pack(side="right")
        self.slider = Slider(f, 50, command=lambda v: self.soc_lbl.config(
            text=f"{v}%", fg=soc_color(v)))
        self.slider.pack(fill="x", pady=(4, 16))

        tk.Label(f, text="Location", font=F(10), fg=C["muted"], bg=C["card"]).pack(anchor="w", pady=(0, 6))
        tk.Label(f, text=f"\u2316  {DATA_LOCATION}, Syria", font=F(13, "bold"), fg=C["text"],
                 bg=C["card"]).pack(anchor="w")
        tk.Label(f, text=f"NASA POWER dataset: {self.app.data.summary()}", font=F(9), fg=C["dim"],
                 bg=C["card"], wraplength=420, justify="left").pack(anchor="w", pady=(2, 0))

        self.insight = tk.Label(f, font=F(10), fg=C["muted"], bg=C["card"], justify="left",
                                anchor="w", wraplength=420)
        self.insight.pack(anchor="w", fill="x", pady=(12, 0))
        btns = tk.Frame(f, bg=C["card"])
        btns.pack(side="bottom", anchor="w", pady=(16, 0))
        PillButton(btns, "Save and recalculate", self.save).pack(side="left")
        PillButton(btns, "Undo changes", self.load, kind="ghost").pack(side="left", padx=10)

        flow = Card(b, fill=True, pad=24)
        flow.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(flow.inner, text="How energy moves through your home", font=F(14, "bold"),
                 fg=C["text"], bg=C["card"]).pack(anchor="w")
        tk.Label(flow.inner, text="Live picture of today with your current settings.",
                 font=F(10), fg=C["muted"], bg=C["card"]).pack(anchor="w", pady=(2, 0))
        self.flow = FlowDiagram(flow.inner, self.app)
        self.flow.pack(fill="both", expand=True, pady=(8, 0))
        self.load()

    def load(self):
        s = self.app.system
        for fld, key in ((self.f_pv, "pv_kw"), (self.f_inv, "inverter_kw"), (self.f_bat, "battery_kwh")):
            fld.var.set(f"{s[key]:g}" if s else "")
            fld.box.set_border(C["border"])
        soc = s["soc"] if s else 50
        self.slider.set(soc)
        self.soc_lbl.config(text=f"{soc}%", fg=soc_color(soc))
        self.refresh()

    def save(self):
        vals, bad = {}, []
        for key, fld, name in (("pv_kw", self.f_pv, "Solar panels"), ("inverter_kw", self.f_inv, "Inverter"),
                               ("battery_kwh", self.f_bat, "Battery")):
            try:
                vals[key] = fld.number()
            except ValueError:
                bad.append(name)
        if bad:
            self.app.toast("Check your numbers", f"{', '.join(bad)}: enter a number above zero, like 3 or 5.5.",
                           "danger")
            return
        vals["soc"] = self.slider.value
        first = not self.app.ready
        self.app.system = vals
        self.app.refresh_all()
        if first and not self.app.appliances:
            self.app.toast("System saved", "Next, add the devices you want to run.", "success")
            self.app.show("appliances")
        else:
            self.app.toast("System saved", "Recommendations now use your new settings.", "success")

    def refresh(self):
        s = self.app.system
        if not s:
            self.insight.config(text="Fill in your system and press Save to start.")
            self.flow.draw()
            return
        notes = [f"Usable energy in the battery right now: {s['battery_kwh'] * s['soc'] / 100:.1f} kWh."]
        if s["inverter_kw"] < s["pv_kw"]:
            notes.append(f"Your inverter is smaller than your panels, so output is capped at "
                         f"{s['inverter_kw']:g} kW on sunny middays.")
        self.insight.config(text="\n".join(notes))
        self.flow.draw()


class AppliancesPage(Page):
    def build(self):
        self.header("Appliances", "The devices you use and how much power each one draws on average")
        b = self.body()
        b.columnconfigure(0, weight=7, uniform="a")
        b.columnconfigure(1, weight=4, uniform="a")
        b.rowconfigure(0, weight=1)

        tcard = Card(b, fill=True, pad=22)
        tcard.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        top = tk.Frame(tcard.inner, bg=C["card"])
        top.pack(fill="x", pady=(0, 14))
        tk.Label(top, text="Your devices", font=F(14, "bold"), fg=C["text"], bg=C["card"]).pack(side="left")
        PillButton(top, "Remove", self.remove, kind="danger", height=36).pack(side="right")
        PillButton(top, "Run today on/off", self.toggle, kind="ghost", height=36).pack(side="right", padx=8)
        cols = ("name", "power", "duration", "priority", "run")
        self.tree = ttk.Treeview(tcard.inner, columns=cols, show="headings", style="SW.Treeview",
                                 selectmode="browse")
        for c, txt, wd, anc in (("name", "Appliance", 200, "w"), ("power", "Power", 90, "center"),
                                ("duration", "Runs for", 90, "center"),
                                ("priority", "Priority", 120, "center"), ("run", "Run today", 100, "center")):
            self.tree.heading(c, text=txt, anchor=anc)
            self.tree.column(c, width=wd, anchor=anc, stretch=(c == "name"))
        self.tree.tag_configure("odd", background=C["card"])
        self.tree.tag_configure("even", background="#142040")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.toggle())
        self.hint = tk.Label(tcard.inner, font=F(9), fg=C["dim"], bg=C["card"])
        self.hint.pack(anchor="w", pady=(10, 0))

        fcard = Card(b, fill=True, pad=22)
        fcard.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        f = fcard.inner
        tk.Label(f, text="Add a device", font=F(14, "bold"), fg=C["text"], bg=C["card"]).pack(anchor="w", pady=(0, 10))
        self.lib_menu = tk.Menu(self, tearoff=0, bg=C["card2"], fg=C["text"], activebackground=C["sun"],
                                activeforeground=C["bg"], bd=0, font=F(10))
        for item in APPLIANCE_LIBRARY:
            self.lib_menu.add_command(label=f"{item['name']}    {item['power']:g} kW, {item['duration']:g} h",
                                      command=lambda it=item: self.pick(it))
        self.lib_btn = PillButton(f, "Choose a common device  \u25be", self.open_library, kind="ghost", height=38)
        self.lib_btn.pack(anchor="w", pady=(0, 12))
        self.f_name = Field(f, "Name", "", "")
        self.f_name.pack(fill="x", pady=(0, 12))
        row = tk.Frame(f, bg=C["card"])
        row.pack(fill="x", pady=(0, 12))
        row.columnconfigure(0, weight=1, uniform="r")
        row.columnconfigure(1, weight=1, uniform="r")
        self.f_power = Field(row, "Average power", "kW")
        self.f_dur = Field(row, "Runs for", "h")
        self.f_power.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.f_dur.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(f, text="Priority", font=F(10), fg=C["muted"], bg=C["card"]).pack(anchor="w", pady=(0, 8))
        self.prio = ChipGroup(f, PRIORITIES, "Flexible", per_row=3, colors=PRIORITY_COLORS)
        self.prio.pack(anchor="w")
        self.run_chip = Chip(f, "\u2713 Run today", selected=True, color=C["green"])
        self.run_chip.command = lambda: self.run_chip.set_selected(not self.run_chip.selected)
        self.run_chip.pack(anchor="w", pady=(4, 14))
        PillButton(f, "Add device", self.add).pack(anchor="w")

        legend = tk.Frame(f, bg=C["card"])
        legend.pack(side="bottom", fill="x")
        for p, txt in (("Essential", "Runs no matter what, like the fridge"),
                       ("Important", "Can wait a few hours"),
                       ("Flexible", "SolarWise moves it to the sunniest time")):
            r = tk.Frame(legend, bg=C["card"])
            r.pack(fill="x", pady=3)
            tk.Label(r, text="\u25cf", fg=PRIORITY_COLORS[p], bg=C["card"], font=F(10)).pack(side="left")
            tk.Label(r, text=f"{p}: {txt}", fg=C["muted"], bg=C["card"], font=F(9)).pack(side="left", padx=6)

    def refresh(self):
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for i, a in enumerate(self.app.appliances):
            self.tree.insert("", "end", iid=str(i), tags=("odd" if i % 2 else "even",), values=(
                a["name"], f"{a['power']:g} kW", f"{a['duration']:g} h", a["priority"],
                "\u25cf  Yes" if a.get("run") else "\u25cb  No"))
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0])
        self.hint.config(text="Double-click a row to switch 'Run today' on or off." if self.app.appliances
                         else "No devices yet. Pick one with 'Choose a common device' on the right.")

    def open_library(self):
        b = self.lib_btn
        self.lib_menu.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())

    def pick(self, item):
        """Fill the form from the team's appliance power table."""
        self.f_name.var.set(item["name"])
        self.f_power.var.set(f"{item['power']:g}")
        self.f_dur.var.set(f"{item['duration']:g}")
        for fld in (self.f_name, self.f_power, self.f_dur):
            fld.box.set_border(C["border"])
        self.prio.select(item["priority"], notify=False)

    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            self.app.toast("Pick a device first", "Click a row in the table, then try again.", "warn")
            return None
        return int(sel[0])

    def toggle(self):
        i = self._selected()
        if i is not None:
            a = self.app.appliances[i]
            a["run"] = not a.get("run")
            self.app.refresh_all()

    def remove(self):
        i = self._selected()
        if i is not None:
            name = self.app.appliances.pop(i)["name"]
            self.app.refresh_all()
            self.app.toast("Device removed", f"{name} is no longer in your list.", "info")

    def add(self):
        name = self.f_name.text()
        if not name:
            self.f_name.box.set_border(C["red"])
            self.app.toast("Name the device", "Type a name like 'Oven' or 'Electric kettle'.", "danger")
            return
        if any(a["name"].lower() == name.lower() for a in self.app.appliances):
            self.app.toast("Already in your list", f"{name} is already added. Edit or remove it first.", "warn")
            return
        try:
            power, dur = self.f_power.number(), self.f_dur.number()
        except ValueError:
            self.app.toast("Check your numbers", "Power and time must be numbers above zero.", "danger")
            return
        self.app.appliances.append({"name": name, "power": power, "duration": dur,
                                    "priority": self.prio.value, "run": self.run_chip.selected})
        self.f_name.var.set("")
        self.app.refresh_all()
        self.app.toast("Device added", f"{name} ({power:g} kW) is in your list.", "success")


class BestTimePage(Page):
    def build(self):
        self.header("Best time to run", "Pick a device and SolarWise finds the sunniest slot for it")
        self.seg = Segmented(self.header_right, ["Today", "Tomorrow"], width=200,
                             command=lambda v: self.app.set_day(v.lower()))
        self.seg.pack(side="right")
        self.datenav = DateNav(self.header_right, self.app, cloudy=False)
        self.datenav.pack(side="right", padx=16)
        b = self.body()
        self.chips = ChipGroup(b, [], command=lambda v: self.refresh(), per_row=7)
        self.chips.pack(anchor="w", pady=(0, 10))

        hero = self.hero = Card(b, pad=28, radius=22, border="#2B6A5C")
        hero.pack(fill="x")
        h = hero.inner
        left = tk.Frame(h, bg=C["card"])
        left.pack(side="left", fill="x", expand=True)
        self.h_label = tk.Label(left, font=F(11), fg=C["muted"], bg=C["card"])
        self.h_label.pack(anchor="w")
        self.h_time = tk.Label(left, font=F(40, "bold"), fg=C["green"], bg=C["card"])
        self.h_time.pack(anchor="w")
        self.h_sub = tk.Label(left, font=F(11), fg=C["text"], bg=C["card"])
        self.h_sub.pack(anchor="w")
        PillButton(h, "\u23f0  Remind me", self.notify, width=160, height=46).pack(side="right", anchor="n")

        stats = Card(b, pad=20)
        stats.pack(fill="x", pady=16)
        self.m = {}
        for i, (k, lab) in enumerate((("solar", "Expected sun power"), ("need", "Device needs"),
                                      ("surplus", "Left over"), ("cover", "Covered by solar"))):
            stats.inner.columnconfigure(i * 2, weight=1)
            if i:
                tk.Frame(stats.inner, bg=C["border"], width=1).grid(row=0, column=i * 2 - 1, sticky="ns", padx=18)
            self.m[k] = Metric(stats.inner, lab)
            self.m[k].grid(row=0, column=i * 2, sticky="w")

        tl = Card(b, pad=22)
        tl.pack(fill="x")
        top = tk.Frame(tl.inner, bg=C["card"])
        top.pack(fill="x")
        tk.Label(top, text="Sun through the day", font=F(13, "bold"), fg=C["text"], bg=C["card"]).pack(side="left")
        tk.Label(top, text="Brighter blocks mean more solar power", font=F(9), fg=C["dim"],
                 bg=C["card"]).pack(side="right")
        self.timeline = Timeline(tl.inner)
        self.timeline.pack(fill="x", pady=(10, 0))
        self.alt_lbl = tk.Label(tl.inner, font=F(10), fg=C["sky"], bg=C["card"])
        self.alt_lbl.pack(anchor="w", pady=(6, 0))

    def current(self):
        r = self.app.res[self.app.day]
        name = self.chips.value
        a = next((x for x in self.app.appliances if x["name"] == name), None)
        return r, a, r["windows"].get(name, (None, None))

    def refresh(self):
        if not self.app.ready:
            return
        self.seg.set(self.app.day.title())
        self.datenav.refresh()
        names = [a["name"] for a in self.app.appliances if a["name"] in self.app.res[self.app.day]["windows"]]
        self.chips.set_options(names, self.chips.value)
        r, a, (best, alt) = self.current()
        if not a or not best:
            self.h_label.config(text="Nothing to schedule yet")
            self.h_time.config(text="--:--", fg=C["dim"])
            self.h_sub.config(text="Add a Flexible or Important device on the Appliances page.")
            return
        ok = best["coverage"] >= .99
        self.hero.set_border("#2B6A5C" if ok else "#6A4A22")
        self.h_label.config(text=f"Best time {self.app.day} ({self.app.shown_date():%d %b %Y}) "
                                 f"for your {a['name'].lower()}")
        self.h_time.config(text=f"{fmt_time(best['start'])} \u2013 {fmt_time(best['end'])}",
                           fg=C["green"] if ok else C["amber"])
        self.h_sub.config(text=(f"Uses {a['power']:g} kW for {a['duration']:g} h, fully on sunshine."
                                if ok else f"Uses {a['power']:g} kW for {a['duration']:g} h. Part of it "
                                           f"will come from the battery."))
        cover = best["coverage"]
        self.m["solar"].set(f"{best['solar']:.1f}", "kW", "average over the slot", C["sun"])
        self.m["need"].set(f"{a['power']:.1f}", "kW", "plus the rest of the house")
        self.m["surplus"].set(f"{best['surplus']:+.1f}", "kW", "extra solar" if ok else "taken from battery",
                              C["green"] if ok else C["red"])
        self.m["cover"].set(f"{cover:.0%}", "", "of this device's energy", C["green"] if cover >= 1 else C["amber"])
        self.timeline.set(r["generation"], best, alt, self.app.day == "today")
        self.alt_lbl.config(text=(f"Alternative: {fmt_time(alt['start'])} \u2013 {fmt_time(alt['end'])}, "
                                  f"{alt['surplus']:+.1f} kW left over") if alt else "")

    def notify(self):
        r, a, (best, _) = self.current()
        if not a or not best:
            return
        span = f"{fmt_time(best['start'])} \u2013 {fmt_time(best['end'])}"
        self.app.add_reminder(a["name"], best, self.app.day)
        self.app.toast("Reminder set", f"We'll notify you at {fmt_time(best['start'])} to run the "
                                       f"{a['name'].lower()}.", "success")
        self.app.after(3500, lambda: self.app.toast(
            f"Time to run the {a['name'].lower()}",
            f"Sun is strong now ({best['solar']:.1f} kW). Best until {fmt_time(best['end'])}. "
            f"(Preview of the real reminder)", "info"))


class AlertsPage(Page):
    def build(self):
        self.header("Notifications", "Weather warnings, reminders you set and battery tips")
        PillButton(self.header_right, "Clear my reminders", self.clear, kind="ghost").pack(side="right")
        self.box = self.body()

    def refresh(self):
        for wdg in self.box.winfo_children():
            wdg.destroy()
        items = self.app.all_alerts()
        if not items:
            tk.Label(self.box, text="No notifications. SolarWise will warn you here before a cloudy day.",
                     font=F(11), fg=C["muted"], bg=C["bg"]).pack(anchor="w", pady=20)
            return
        for it in items[:7]:
            col, sym = KIND[it["kind"]]
            card = Card(self.box, pad=16, border=lerp(col, C["card"], .7) if it["kind"] == "danger" else None)
            card.pack(fill="x", pady=(0, 10))
            inn = card.inner
            Badge(inn, sym, col, size=42).pack(side="left", anchor="n")
            tx = tk.Frame(inn, bg=C["card"])
            tx.pack(side="left", fill="x", expand=True, padx=14)
            tk.Label(tx, text=it["title"], font=F(12, "bold"), fg=C["text"], bg=C["card"]).pack(anchor="w")
            tk.Label(tx, text=it["text"], font=F(10), fg=C["muted"], bg=C["card"], justify="left",
                     wraplength=720).pack(anchor="w", pady=(2, 0))
            tk.Label(inn, text=it["when"], font=F(9), fg=C["dim"], bg=C["card"]).pack(side="right", anchor="n")

    def clear(self):
        self.app.reminders.clear()
        self.app.refresh_all()


class InsightsPage(Page):
    """Findings from the whole dataset: averages, best/worst month, cloudy days,
    change over the years, and normal vs weather-based scheduling."""

    HABITS = ["08:00", "13:00", "18:00", "21:00"]

    def build(self):
        self.header("Data insights", "")
        b = self.body()
        b.columnconfigure(0, weight=3, uniform="i")
        b.columnconfigure(1, weight=2, uniform="i")
        b.rowconfigure(1, weight=1)

        strip = Card(b, pad=20)
        strip.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        self.m = {}
        for i, (k, lab) in enumerate((("sun", "Average sunshine"), ("best", "Best month for your panels"),
                                      ("worst", "Weakest month"), ("cloudy", "Cloudy days a year"))):
            strip.inner.columnconfigure(i * 2, weight=1)
            if i:
                tk.Frame(strip.inner, bg=C["border"], width=1).grid(row=0, column=i * 2 - 1, sticky="ns", padx=18)
            self.m[k] = Metric(strip.inner, lab)
            self.m[k].grid(row=0, column=i * 2, sticky="w")

        left = Card(b, fill=True, pad=22)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(left.inner, text="Your system's average output by month", font=F(13, "bold"),
                 fg=C["text"], bg=C["card"]).pack(anchor="w")
        tk.Label(left.inner, text="kWh per day. The cloud figure is the share of cloudy days in that month.",
                 font=F(9), fg=C["muted"], bg=C["card"]).pack(anchor="w", pady=(2, 8))
        self.note = auto_wrap(tk.Label(left.inner, font=F(10), fg=C["muted"], bg=C["card"], justify="left",
                                       anchor="w"))
        self.note.pack(side="bottom", anchor="w", fill="x", pady=(10, 0))
        self.chart = BarChart(left.inner)
        self.chart.pack(fill="both", expand=True)

        right = Card(b, fill=True, pad=22)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        r = right.inner
        tk.Label(r, text="Normal vs weather-based scheduling", font=F(13, "bold"), fg=C["text"],
                 bg=C["card"]).pack(anchor="w")
        self.cmp_sub = auto_wrap(tk.Label(r, font=F(9), fg=C["muted"], bg=C["card"], justify="left", anchor="w"))
        self.cmp_sub.pack(anchor="w", fill="x", pady=(2, 14))
        tk.Label(r, text="When you usually run your devices", font=F(10), fg=C["muted"],
                 bg=C["card"]).pack(anchor="w", pady=(0, 8))
        self.habit = ChipGroup(r, self.HABITS, "18:00", command=self.set_habit, per_row=4)
        self.habit.pack(anchor="w", pady=(0, 10))
        self.bars = CompareBars(r)
        self.bars.pack(fill="x")
        self.cmp_note = auto_wrap(tk.Label(r, font=F(10), fg=C["text"], bg=C["card"], justify="left",
                                           anchor="w"))
        self.cmp_note.pack(anchor="w", fill="x", pady=(6, 0))

    def set_habit(self, value):
        self.app.habit_hour = float(value[:2])
        self.app.insights = historical_insights(self.app.data, self.app.system, self.app.appliances,
                                                self.app.habit_hour)
        self.refresh()

    def refresh(self):
        if not self.app.ready:
            return
        ins = self.app.insights
        y0, y1 = ins["years"]
        self.sub_lbl.config(text=f"What {y1 - y0 + 1} years of NASA POWER hourly data say about solar in "
                                 f"{DATA_LOCATION} ({self.app.data.summary()})")
        self.m["sun"].set(f"{ins['avg_sun']:.1f}", "kWh/m\u00b2",
                          f"about {ins['avg_kwh']:.1f} kWh a day from your panels", C["sun"])
        self.m["best"].set(ins["best_month"], "", f"{ins['best_kwh']:.1f} kWh a day on average", C["sun"])
        self.m["worst"].set(ins["worst_month"], "", f"{ins['worst_kwh']:.1f} kWh a day on average", C["sky"])
        self.m["cloudy"].set(f"{ins['cloudy_days']:.0f}", "days",
                             f"under {LOW_CLEARNESS:.0%} of clear-sky sunshine")
        self.chart.set(MONTHS, ins["monthly_kwh"], ins["monthly_cloudy"])
        yk = ins["year_kwh"]
        notes = [f"Expected in a typical year: about {np.mean(yk):,.0f} kWh from your panels "
                 f"(between {min(yk):,.0f} and {max(yk):,.0f} in past years)." if yk else "",
                 f"Yearly sunshine changed by {ins['trend_pct']:+.1f}% between {y0} and {y1}.",
                 f"Rainy days (1 mm or more) get about {ins['rain_drop_pct']:.0f}% less sunshine than dry days."]
        if not math.isnan(ins["temp_sun_corr"]):
            notes.append(f"Hot days are also sunny days (correlation {ins['temp_sun_corr']:.2f}), "
                         f"but heat lowers panel efficiency.")
        self.note.config(text="  ".join(notes))

        cmp = ins["compare"]
        self.habit.select(f"{int(self.app.habit_hour):02d}:00", notify=False)
        if not cmp:
            self.cmp_sub.config(text="Uses the devices you plan to run.")
            self.bars.set([])
            self.cmp_note.config(text="Add Flexible or Important devices with 'Run today' switched on "
                                      "to see this comparison.", fg=C["muted"])
            return
        self.cmp_sub.config(text=f"Simulated over every day of {cmp['year']} for the devices you plan to run. "
                                 f"Share of their energy that comes straight from the sun:")
        self.bars.set([(f"Same time every day ({fmt_time(cmp['habit'])})", cmp["normal_pct"], C["red"]),
                       ("Following SolarWise", cmp["smart_pct"], C["green"])])
        moved = cmp["smart_kwh"] - cmp["normal_kwh"]
        self.cmp_note.config(fg=C["text"], text=f"About {moved:,.0f} kWh a year moves from the battery or grid "
                                                f"onto direct sunshine.")


# SolarWise logo for the window / taskbar icon (PNG, base64) - replaces Tk's feather
APP_ICON = {
    16: (
        "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAACN0lEQVR42o2Tz0vUURTFP/e9+c74nRnHnFGCwoJyTFuIEkZR"
        "mzaRQkS7FuWiVbhq07JdmwhaRf9Ergr6sW1RIEEYUYQjBVGKFhOj8/v7vbeFjlmIduBt7uWce+557wlAurc4Jc7dA0bAAISd"
        "YZutj6Z6q1ZeeCrp3qFJ8fIExGNqu5D/iIgTsNhiuyiZfHEB5wYxiwG/E8MJiECsW6UYEY9qSTKFIduw9vdk5zoFodZUosjI"
        "hA4RMOusg7h/yd4JarC2rlTWlEolYvhgwNREBjVoR4Z0lMES28lOhMp6RD6X4PxYhjAQekLH3WsFevYHzL6oMP1gpeMAQDor"
        "4ASq9Yjpc/1cPpOksqYsrUbMXOgh0+2orythn+fS7e88nquSyzpihcSGjNBqNxkdO0s1UWfm4TzfVoFGTJAUbl4tEAZC9ZdS"
        "Wm4RBLLlIiEiRO0mA4eK5LrTPHr+knQ6JL8PTD13ZsuEKcfIAMy+O0JZI5J+ESQFZhsOGpEjky/yau41qKNWi6lVN9NR48b9"
        "ZSCCXJPxE6dY+ryID5UwJSRazYjTI3kmT7Y5nlK8z29eDFtROweGYHHE0eEG4/kDvP1U5sPXiISZWCHdkmx7nonBELfHO2zX"
        "5xk9HLDyI8n7L22TbN8xqzbUrGmC2PbhO0ME1JAusUyXl4SqlrKhG3Rpi0E8/wWL1cTHsZZ80NVXQuSKmvOqZmqIGuxyTHHO"
        "IEaZ9u3Gz1KQyr8RkXGQ/m3Z7fWdr9fKC89+A+UwAF0BSytLAAAAAElFTkSuQmCC"
    ),
    32: (
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAEwElEQVR42sWXS2xUZRTHf+f77jw6M31OgbYqkYpDgPAIBAM+"
        "0pgqoAs1xrLQyIJE3RujCxfERBdu3JgYE42ujCGEBS4wYgm6wkeIpPERS2l5Q59YOp3n/b7j4k5bE6EWKeWfTO7NLO55/c//"
        "nCNEMIBPL1+1gjD+OoZnUF0DCIsDReRPPF8RVD6YHhkanrEpMy+pltU7RewnImal4kGVRYUIgkHVn1d1rxQmBo4CRgDS2Vy3"
        "iOlVFNQ7ELOI0c9lAfWIsYKg6p+YHu8/JpnM6mUaN6cQaQf1gOXOIgpQ9YpU/GYby2T3i7G7l8j4LN/EmEY1qgbo0ajgwtJB"
        "ajZ7JJ3NKXcRQUSO249eAGNAREAVpwtqJA0Ww7gRcAr5vAevIEIiKSRigvPz+x38n0jFzHktIhTLnngg9DyWYVNnggsjIUd+"
        "nubiNUemTvB+/hIsCNZEz2oIxbIHX/PGQ3tLwKG32tixKRXlXYQrVyv0vD/Mif4SmbqbZ+I/SSgiCMpU0YODlibL+vvipJMG"
        "KzA26XjvxRa6H85QnQipUYBYxjI2FrL5jQuMTTligdyQE8H8tRXKVUfVCU9vSfPCIxm61ibpbItBolaEioJT/GRkZAZh3tG6"
        "IuDZbWk+OjJJskEI3S04YIyhXK7Qls3w6WsNPLk1A/UWQsVNOP66FgLQlLYYewMmS1SlzhWxqM+QWsMtwAERQ6VcIptt46Gu"
        "51i/5jTYc3x9vMjBE1P8cqbM6HVHqapsfzDJ4bfbwEcSN+uICkaEwZFq7U9dGAdEhDCs0tS8jK7HnydfFoqXjlNvLnD4xzI4"
        "xcaFwIIIlIpK7/4Ouh/NUB2f4YAQq1OGx6psefMy43m/cA549cRicbZt342NJTj1/QEuXb4Apo76OsEYg/caqZeAJIWXPxzh"
        "UMrUugBQx2Qhw7vf3ct4YZSYLd9UlIJ/p75AbsMOWpd1cPKnXkaGL9FQXw/qcB68m/uSKgQWxvOOne9c5qmtKTY/UMfgxSlO"
        "59dxz9qNrFt7lV/7fiCRSKHq5y+BqiceT9K96yVKpWmOf3sgktYFKmGhMKMPSjxp2LV7D/FEhmPffEGlUkLE3HA0zkaPD2nv"
        "WElTYxMXz/6G+gqxwBJYmfdnjBC3QkujJdtiaW2OE0iF80O/09zYRHvHSvAhscAQWLlxCVShcD1E421MFAz954YpTkGxVL0Z"
        "gedRLwdV6D83TOcGg8bbKEz2UUg4UCWVNswkVtLZnHqFVEJ4tTvDmtw6kok0g2f6CMNqbbrd+sBQVYIgRufqjZSK05wd+gMb"
        "WEoVz8dHr1OoKEZAMq05rYbK8kbL+c9XQZ2DioNEIqK53sZ8VoVKOZrTQRwCgUnHyn1DjNSUU9LZnKpCLIAtqxLEgpqcqEcX"
        "YUcQMSiKegUB55STQ2WqYa2N09lcxFuFQtnDnd6PBFKJWQ7MLSQi0JAyS7KG/WM0S4DqEMbcj6o6z9J4AB4RwfuzBjgoUUKW"
        "cjnVms2Dd/0wMfn8wCjCXhERRCyou0PZUFCHiI1ssTefHxi1gKkWxweDZNMJwXSJ2GbkDhwpIiJiDarnVd2ewsTpXsDYWrSm"
        "WpwYiNc3fIkzHiELZBf/PNfPCCr7CmODfTMn2t+pShLRj5W7NAAAAABJRU5ErkJggg=="
    ),
    48: (
        "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"
    ),
    64: (
        "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"
    ),
}


# ------------------------------------------------------------------- app --
class SolarWiseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SolarWise  |  Smart Solar Energy Advisor")
        self.geometry("1380x900")
        self.minsize(1240, 840)
        self.configure(bg=C["bg"])
        try:   # project logo instead of the default feather
            self._icons = [tk.PhotoImage(data=APP_ICON[k]) for k in sorted(APP_ICON, reverse=True)]
            self.iconphoto(True, *self._icons)
        except tk.TclError:
            pass
        try:
            self.data = load_data(self)
        except SystemExit:
            self.destroy()
            raise
        except Exception as e:
            messagebox.showerror("SolarWise", f"Could not read the dataset:\n{e}", parent=self)
            self.destroy()
            raise SystemExit(1)
        self.system = None          # filled in by the user on "My system"
        self.appliances = []        # filled in by the user on "Appliances"
        self.sim_date = datetime.date.today()   # the day treated as "today"
        self.day = "tomorrow"
        self.habit_hour = 18.0
        self.reminders, self.toasts, self.fired = [], [], set()
        self._style()
        self.recompute()
        self._build()
        self.show("system")
        self.after(900, self._startup_alert)
        self.after(20000, self._tick)

    # -- data
    @property
    def ready(self):
        return self.system is not None

    def recompute(self):
        """Run the pandas analysis for the chosen day and the next one."""
        if not self.ready:
            self.res, self.warning, self.insights = None, None, None
            return
        self.res = {d: analyze_day(self.data, self.system, self.appliances, self.shown_date(d))
                    for d in ("today", "tomorrow")}
        self.warning = tomorrow_warning(self.res["today"], self.res["tomorrow"])
        self.insights = historical_insights(self.data, self.system, self.appliances, self.habit_hour)

    def shown_date(self, day=None):
        return self.sim_date + datetime.timedelta(days=1 if (day or self.day) == "tomorrow" else 0)

    def set_date(self, d):
        """Show the typed date (as 'today' or 'tomorrow', whichever is selected)."""
        lo, hi = datetime.date(1990, 1, 1), datetime.date(2100, 12, 30)
        d = min(max(d, lo), hi)
        self.sim_date = d - datetime.timedelta(days=1 if self.day == "tomorrow" else 0)
        self.refresh_all()

    def shift_date(self, n):
        self.set_date(self.shown_date() + datetime.timedelta(days=n))

    def go_today(self):
        self.day = "today"
        self.set_date(datetime.date.today())

    def jump_cloudy(self):
        """Inside the dataset: next sunny day followed by a cloudy one.
        Outside it: next day whose tomorrow has a high chance of cloud."""
        if self.data.is_expected(self.sim_date + datetime.timedelta(days=1)):
            d = self.sim_date + datetime.timedelta(days=1)
            for _ in range(366):
                if self.data.expected_stats(d + datetime.timedelta(days=1))["cloud_chance"] >= CLOUD_CHANCE_ALERT:
                    break
                d += datetime.timedelta(days=1)
            self.sim_date, self.day = d, "tomorrow"
            self.refresh_all()
            if self.warning:
                self.toast(self.warning["title"], self.warning["text"], "warn")
            return
        cl = self.data.daily["clearness"]
        nxt = cl.shift(-1)
        consecutive = (cl.index.to_series().shift(-1) - cl.index.to_series()) == pd.Timedelta(days=1)
        hits = cl.index[(cl >= .7) & (nxt < LOW_CLEARNESS) & consecutive]
        dates = [d.date() for d in hits]
        later = [d for d in dates if d > self.sim_date]
        if not dates:
            self.toast("No cloudy days found", "The dataset has no sunny-then-cloudy pair.", "info")
            return
        self.sim_date, self.day = (later or dates)[0], "tomorrow"
        self.refresh_all()
        if self.warning:
            self.toast(self.warning["title"], self.warning["text"], "danger")

    def focus_window(self, day):
        """(name, window) of the most important device the user plans to run."""
        if not self.ready:
            return None
        w = self.res[day]["windows"]
        runs = [a for a in self.appliances if a.get("run") and a["name"] in w and w[a["name"]][0]]
        runs.sort(key=lambda a: (a["priority"] != "Flexible", -a["power"]))
        return (runs[0]["name"], w[runs[0]["name"]][0]) if runs else None

    def all_alerts(self):
        if not self.ready:
            return []
        items = [dict(r, when="Set by you") for r in reversed(self.reminders)]
        if self.warning:
            items.append(dict(self.warning, when="Tomorrow"))
        for rec in self.res["tomorrow"]["recommendations"]:
            if rec["kind"] in ("success", "warn") and ":" in rec["title"]:
                items.append(dict(kind=rec["kind"], title=f"Tomorrow, {rec['title']}", text=rec["text"],
                                  when="Tomorrow"))
        full = self.res["today"]["battery_full_at"]
        if full:
            items.append(dict(kind="info", title=f"Battery should be full by {full:02d}:00 today",
                              text="Use heavy appliances after that so no sunshine goes to waste.", when="Today"))
        return items

    def add_reminder(self, name, win, day):
        date = datetime.date.today() + datetime.timedelta(days=1 if day == "tomorrow" else 0)
        at = datetime.datetime.combine(date, datetime.time()) + datetime.timedelta(hours=win["start"])
        self.reminders.append({"kind": "success", "at": at, "name": name,
                               "title": f"Reminder: {name} at {fmt_time(win['start'])} {day}",
                               "text": f"Best slot {fmt_time(win['start'])} \u2013 {fmt_time(win['end'])} "
                                       f"with about {win['solar']:.1f} kW of sun."})
        self.update_sidebar()

    # -- ui
    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("SW.Treeview", background=C["card"], fieldbackground=C["card"], foreground=C["text"],
                     rowheight=40, borderwidth=0, font=F(11))
        st.configure("SW.Treeview.Heading", background=C["card2"], foreground=C["muted"], relief="flat",
                     font=F(10, "bold"), padding=(10, 9), borderwidth=0)
        st.map("SW.Treeview", background=[("selected", "#3A2F1C")], foreground=[("selected", C["sun"])])
        st.map("SW.Treeview.Heading", background=[("active", C["border"])])
        st.layout("SW.Treeview", [("SW.Treeview.treearea", {"sticky": "nswe"})])

    def _build(self):
        side = tk.Frame(self, bg=C["side"], width=248)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Frame(self, bg=C["border"], width=1).pack(side="left", fill="y")

        logo = tk.Canvas(side, height=96, bg=C["side"], highlightthickness=0)
        logo.pack(fill="x")
        cx, cy = 42, 56
        logo.create_arc(cx - 22, cy - 22, cx + 22, cy + 22, start=0, extent=180, style="arc",
                        outline=C["dim"], dash=(2, 3))
        logo.create_line(cx - 28, cy, cx + 28, cy, fill=C["sun"], width=2)
        logo.create_oval(cx + 2, cy - 26, cx + 16, cy - 12, fill=C["sun"], outline="")
        logo.create_text(80, 42, text="SolarWise", anchor="w", fill=C["text"], font=F(17, "bold"))
        logo.create_text(81, 64, text="Solar first, battery last", anchor="w", fill=C["muted"], font=F(9))

        self.nav = {}
        for key, icon, text in (("dashboard", "\u25d0", "Dashboard"), ("system", "\u2699", "My system"),
                                ("appliances", "\u26a1", "Appliances"), ("besttime", "\u25f7", "Best time"),
                                ("insights", "\u25a4", "Data insights"), ("alerts", "\u2691", "Notifications")):
            item = NavItem(side, icon, text, lambda k=key: self.show(k))
            item.pack(fill="x", padx=12, pady=2)
            self.nav[key] = item

        foot = tk.Frame(side, bg=C["side"])
        foot.pack(side="bottom", fill="x", padx=16, pady=18)
        bcard = Card(foot, pad=14, color=C["card"])
        bcard.pack(fill="x")
        tk.Label(bcard.inner, text="Battery now", font=F(9), fg=C["muted"], bg=C["card"]).pack(anchor="w")
        self.side_bat = tk.Canvas(bcard.inner, height=30, bg=C["card"], highlightthickness=0)
        self.side_bat.pack(fill="x", pady=(6, 0))
        self.side_bat.bind("<Configure>", lambda e: self.update_sidebar())
        self.side_loc = tk.Label(foot, font=F(9), fg=C["muted"], bg=C["side"])
        self.side_loc.pack(anchor="w", pady=(12, 0))
        tk.Label(foot, text="AI4Climate hackathon, Team 7", font=F(8), fg=C["dim"],
                 bg=C["side"]).pack(anchor="w", pady=(2, 0))

        content = tk.Frame(self, bg=C["bg"])
        content.pack(side="left", fill="both", expand=True, padx=28, pady=22)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        self.pages = {}
        for key, cls in (("dashboard", Dashboard), ("system", SystemPage), ("appliances", AppliancesPage),
                         ("besttime", BestTimePage), ("insights", InsightsPage), ("alerts", AlertsPage)):
            p = cls(content, self)
            p.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = p
        for p in self.pages.values():
            p.refresh()
        self.update_sidebar()

    def update_sidebar(self):
        s = self.system
        cv = self.side_bat
        cv.delete("all")
        w = cv.winfo_width()
        if w > 40 and not s:
            cv.create_text(2, 15, text="Set up your system", anchor="w", fill=C["dim"], font=F(10))
        elif w > 40:
            col = soc_color(s["soc"])
            round_rect(cv, 1, 4, w - 60, 26, 7, fill=C["card2"], outline=C["border"])
            cv.create_rectangle(w - 60, 11, w - 56, 19, fill=C["border"], outline="")
            fill_w = (w - 66) * s["soc"] / 100
            if fill_w > 6:
                round_rect(cv, 4, 7, 4 + fill_w, 23, 5, fill=col, outline="")
            cv.create_text(w - 2, 15, text=f"{s['soc']}%", anchor="e", fill=C["text"], font=F(11, "bold"))
        self.side_loc.config(text=f"\u2316  {DATA_LOCATION}, Syria")
        self.nav["alerts"].set_badge(len(self.all_alerts()))

    def show(self, key):
        if key in ("dashboard", "besttime", "insights", "alerts") and not self.ready:
            if key != "system":
                self.toast("Set up your system first", "Enter your panels, inverter and battery, then press Save.",
                           "warn")
            key = "system"
        for k, item in self.nav.items():
            item.set_active(k == key)
        self.pages[key].tkraise()
        self.pages[key].refresh()

    def set_day(self, day):
        self.day = day
        self.pages["dashboard"].refresh()
        self.pages["besttime"].refresh()
        self.update_sidebar()

    def refresh_all(self):
        self.recompute()
        for p in self.pages.values():
            p.refresh()
        self.update_sidebar()

    def toast(self, title, text, kind="info"):
        t = Toast(self, title, text, kind)
        self.toasts.append(t)
        t.place_me()

    def _startup_alert(self):
        if not self.ready:
            self.toast("Welcome to SolarWise", "Start by entering your solar system details.", "info")
        elif self.warning:
            self.toast(self.warning["title"], self.warning["text"], "danger")

    def _tick(self):
        """Every 20 s: fire reminders whose time has come, and today's best-slot alerts."""
        now = datetime.datetime.now()
        for r in self.reminders:
            if not r.get("fired") and now >= r["at"]:
                r["fired"] = True
                self.toast(f"Time to run the {r['name'].lower()}", r["text"], "success")
        h = now.hour + now.minute / 60
        for a in (self.appliances if self.ready else []):
            best = self.res["today"]["windows"].get(a["name"], (None, None))[0]
            key = (now.date(), a["name"])
            if a.get("run") and best and key not in self.fired and 0 <= h - best["start"] < .1:
                self.fired.add(key)
                self.toast(f"Good time for the {a['name'].lower()}",
                           f"About {best['solar']:.1f} kW of sun until {fmt_time(best['end'])}.", "success")
        self.after(20000, self._tick)


if __name__ == "__main__":
    SolarWiseApp().mainloop()
