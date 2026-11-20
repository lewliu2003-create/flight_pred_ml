"""Pull hourly Open-Meteo weather for every airport in the flights file and
merge it onto each flight: origin weather at the scheduled departure hour,
destination weather at the scheduled arrival hour (both in local time).

Same approach as the Colab notebook, runnable as a plain script:
    pip install pandas requests timezonefinder pyarrow
    python flight_api.py
"""

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests
from timezonefinder import TimezoneFinder

# ---- Change these ----
DATA_FILE = "flights_2026_stacked.csv"
CODES_FILE = "Airport Codes.csv"
OUTPUT_FILE = "flights_with_weather.csv"   # ends in .parquet or .csv
CACHE_DIR = Path("weather_cache")              # one parquet file per airport
DROP_LEAKAGE = True     # drop columns only known after the flight happens
DROP_CANCELLED = False  # cancelled flights have no DEP_DEL15 (blank target)
DROP_DIVERTED = False   # diverted flights' dest_* weather is for an airport they never reached
DROP_TERRITORIES = False  # PR, VI, TT (Guam/Saipan); TT arrival dates can be off (date line)
ASK_VARIABLES = True    # False = use DEFAULT_HOURLY without prompting
PAUSE = 2               # seconds between API requests
MAX_RETRIES = 3

LEAK = ["DEP_TIME", "ARR_TIME", "TAXI_OUT", "TAXI_IN", "WHEELS_OFF", "WHEELS_ON",
        "ACTUAL_ELAPSED_TIME", "AIR_TIME", "FLIGHTS"]
TERRITORIES = ["PR", "VI", "TT"]

# Same 10 variables as the Colab notebook
DEFAULT_HOURLY = ["precipitation", "snowfall", "snow_depth", "temperature_2m",
                  "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
                  "cloud_cover", "weather_code", "pressure_msl"]

HOURLY_OPTIONS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "pressure_msl", "surface_pressure", "precipitation", "rain", "snowfall",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "direct_radiation", "direct_normal_irradiance",
    "diffuse_radiation", "sunshine_duration", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "weather_code", "snow_depth",
    "vapour_pressure_deficit", "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]


class FlightAPI:
    URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, codes_csv, out_dir, start_date, end_date, hourly):
        raw = pd.read_csv(codes_csv).dropna(subset=["AIRPORT", "LATITUDE", "LONGITUDE"])
        self.coords = raw.groupby("AIRPORT")[["LATITUDE", "LONGITUDE"]].median()
        tf = TimezoneFinder()
        self.coords["tz"] = [tf.timezone_at(lat=la, lng=lo)
                             for la, lo in zip(self.coords["LATITUDE"],
                                               self.coords["LONGITUDE"])]
        self.hourly = list(hourly)
        # Separate cache folder per date range + variable set, so changing
        # either never mixes old and new files.
        key = hashlib.md5(",".join(sorted(self.hourly)).encode()).hexdigest()[:8]
        self.out_dir = Path(out_dir) / f"{start_date}_{end_date}_{key}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.start_date, self.end_date = start_date, end_date
        self.failed = []

    def build_params(self, lat, lon):
        return {"latitude": lat, "longitude": lon,
                "start_date": self.start_date, "end_date": self.end_date,
                "hourly": ",".join(self.hourly),
                "timezone": "GMT", "wind_speed_unit": "kn"}

    def fetch_airport(self, code):
        out_file = self.out_dir / f"{code}.parquet"
        if out_file.exists():
            return "skipped"
        r = self.coords.loc[code]
        if r["tz"] is None:
            self.failed.append((code, "no time zone found"))
            return "failed"
        try:
            resp = requests.get(self.URL, params=self.build_params(r["LATITUDE"], r["LONGITUDE"]),
                                timeout=60)
            if resp.status_code == 429:
                self.failed.append((code, "429 rate limit"))
                time.sleep(60)
                return "rate_limited"
            resp.raise_for_status()
            w = pd.DataFrame(resp.json()["hourly"])
            # GMT -> airport's local wall-clock time (matches BTS scheduled times)
            w["time"] = (pd.to_datetime(w["time"]).dt.tz_localize("UTC")
                         .dt.tz_convert(r["tz"]).dt.tz_localize(None))
            w["airport"] = code
            w.to_parquet(out_file, index=False)
            return "done"
        except Exception as e:
            self.failed.append((code, str(e)))
            return "failed"

    def fetch_all(self, codes):
        todo = list(codes)
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                wait = 10 * 2 ** (attempt - 1)   # 10s, 20s, 40s
                print(f"Retry {attempt}/{MAX_RETRIES} for {len(todo)} airports in {wait}s...")
                time.sleep(wait)
            self.failed = []
            for i, code in enumerate(todo, 1):
                status = self.fetch_airport(code)
                if status != "skipped":
                    print(f"[{i}/{len(todo)}] {code} {status}", flush=True)
                    time.sleep(PAUSE)
            todo = [c for c, _ in self.failed]
            if not todo:
                break
        if self.failed:
            print("Gave up on:", self.failed)

    def load(self):
        files = list(self.out_dir.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"No weather files in {self.out_dir}")
        w = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        # When clocks fall back in November, one local hour occurs twice.
        # Keep the first so the merge can't duplicate flights.
        return w.drop_duplicates(["airport", "time"], keep="first")


def load_flights(path):
    flights = pd.read_csv(path)
    if DROP_LEAKAGE:
        flights = flights.drop(columns=LEAK, errors="ignore")
    for col in ["CRS_DEP_TIME", "CRS_ARR_TIME"]:
        # Works whether the column was read as int, float or string
        flights[col] = pd.to_numeric(flights[col]).astype(int).astype(str).str.zfill(4)
    flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    if DROP_CANCELLED:
        flights = flights[flights["CANCELLED"] == 0]
    if DROP_DIVERTED:
        flights = flights[flights["DIVERTED"] == 0]
    if DROP_TERRITORIES:
        flights = flights[~flights["ORIGIN_STATE_ABR"].isin(TERRITORIES)
                          & ~flights["DEST_STATE_ABR"].isin(TERRITORIES)]
    return flights.reset_index(drop=True)


def add_datetimes(flights):
    dep_hour = flights["CRS_DEP_TIME"].str[:2].astype(int)
    arr_hour = flights["CRS_ARR_TIME"].str[:2].astype(int)
    dep_min = dep_hour * 60 + flights["CRS_DEP_TIME"].str[2:].astype(int)
    arr_min = arr_hour * 60 + flights["CRS_ARR_TIME"].str[2:].astype(int)
    # Arrival clock time 3+ hours "before" departure = lands the next day (red-eyes)
    next_day = (dep_min - arr_min) > 180
    flights["dep_datetime"] = flights["FL_DATE"] + pd.to_timedelta(dep_hour, unit="h")
    flights["arr_datetime"] = (flights["FL_DATE"] + pd.to_timedelta(arr_hour, unit="h")
                               + pd.to_timedelta(next_day.astype(int), unit="D"))
    print(f"{next_day.sum():,} next-day arrivals")
    return flights


def merge_weather(flights, weather):
    wx_cols = [c for c in weather.columns if c not in ["time", "airport"]]
    origin_wx = weather.rename(columns={**{c: f"origin_{c}" for c in wx_cols},
                                        "airport": "ORIGIN", "time": "dep_datetime"})
    dest_wx = weather.rename(columns={**{c: f"dest_{c}" for c in wx_cols},
                                      "airport": "DEST", "time": "arr_datetime"})
    merged = (flights.merge(origin_wx, on=["ORIGIN", "dep_datetime"], how="left")
                     .merge(dest_wx, on=["DEST", "arr_datetime"], how="left"))
    assert len(merged) == len(flights), "row count changed"
    return merged


def ask_variables(options, default):
    menu = "\n".join(f"{i}: {name}" for i, name in enumerate(options, 1))
    print(f"Which hourly variables do you want?\n{menu}\n"
          f"a: all\nEnter (blank): the default 10\n0: done")
    chosen = []
    while True:
        answer = input().strip().lower()
        if answer == "" and not chosen:
            return list(default)
        if answer in ("0", ""):
            break
        if answer in ("a", "all"):
            return list(options)
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            name = options[int(answer) - 1]
            if name not in chosen:
                chosen.append(name)
        else:
            print("Not a valid choice, try again.")
    return chosen or list(default)


if __name__ == "__main__":
    hourly = ask_variables(HOURLY_OPTIONS, DEFAULT_HOURLY) if ASK_VARIABLES else DEFAULT_HOURLY
    print("Variables:", hourly)

    flights = add_datetimes(load_flights(DATA_FILE))
    print(f"{len(flights):,} flights after filtering "
          f"({int(flights['CANCELLED'].sum()):,} cancelled, {int(flights['DIVERTED'].sum()):,} diverted)")

    # Date range from the data; +1 day so overnight arrivals on the last day get weather
    start = flights["FL_DATE"].min().strftime("%Y-%m-%d")
    end = (flights["FL_DATE"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    api = FlightAPI(CODES_FILE, CACHE_DIR, start, end, hourly)

    codes = pd.concat([flights["ORIGIN"], flights["DEST"]]).unique()
    missing = [c for c in codes if c not in api.coords.index]
    print(f"{len(codes)} airports | missing from codes file: {missing}")

    api.fetch_all([c for c in codes if c in api.coords.index])

    merged = merge_weather(flights, api.load())
    nan_rate = merged.filter(regex="^(origin|dest)_").isna().mean().max()
    print(f"Max missing-weather rate across columns: {nan_rate:.4%}")

    if OUTPUT_FILE.endswith(".csv"):
        merged.to_csv(OUTPUT_FILE, index=False)
    else:
        merged.to_parquet(OUTPUT_FILE, index=False)
    print("Saved", OUTPUT_FILE)
