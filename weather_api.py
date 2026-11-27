"""Pull hourly Open-Meteo weather for every airport in the flights file and
merge it onto each flight: origin weather at the scheduled departure hour,
destination weather at the scheduled arrival hour (both in local time).

LEAD_DAYS chooses which weather:
    0    observed weather (what actually happened)
    1-7  the forecast that was made that many days before each hour

    pip install pandas requests timezonefinder pyarrow
    python flight_weather_api.py
"""

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests
from timezonefinder import TimezoneFinder

# ---- Change these ----
DATA_FILE = "flights_2026_stacked.csv"     # or flights_data.csv, to keep the plane columns
CODES_FILE = "Airport Codes.csv"
LEAD_DAYS = 1                              # 0 = observed, 1-7 = forecast made that many days ahead
OUTPUT_FILE = "flights_with_weather_day1.csv"   # ends in .parquet or .csv
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


def error_text(resp):
    """The API explains a rejected request in a "reason" field; show it."""
    try:
        return f"{resp.status_code}: {resp.json().get('reason', resp.text[:200])}"
    except ValueError:
        return f"{resp.status_code}: {resp.text[:200]}"


class FlightAPI:
    OBSERVED_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

    def __init__(self, codes_csv, out_dir, start_date, end_date, hourly, lead_days=0):
        if lead_days not in range(8):
            raise ValueError("lead_days must be 0 (observed) or 1 to 7")
        raw = pd.read_csv(codes_csv).dropna(subset=["AIRPORT", "LATITUDE", "LONGITUDE"])
        self.coords = raw.groupby("AIRPORT")[["LATITUDE", "LONGITUDE"]].median()
        tf = TimezoneFinder()
        self.coords["tz"] = [tf.timezone_at(lat=la, lng=lo)
                             for la, lo in zip(self.coords["LATITUDE"],
                                               self.coords["LONGITUDE"])]
        self.hourly = list(hourly)
        self.lead_days = lead_days
        self.url = self.FORECAST_URL if lead_days else self.OBSERVED_URL
        # The forecast archive names each variable with its lead time, e.g.
        # temperature_2m_previous_day1 = the value predicted 1 day earlier.
        self.suffix = f"_previous_day{lead_days}" if lead_days else ""
        self.cache_dir = Path(out_dir)
        self.start_date, self.end_date = start_date, end_date
        self.failed = []

    @property
    def out_dir(self):
        # Separate cache folder per date range + lead time + variable set, so
        # changing any of them never mixes old and new files.
        key = hashlib.md5(",".join(sorted(self.hourly)).encode()).hexdigest()[:8]
        lead = f"_day{self.lead_days}" if self.lead_days else ""
        return self.cache_dir / f"{self.start_date}_{self.end_date}{lead}_{key}"

    def build_params(self, lat, lon, variables=None, start_date=None, end_date=None):
        variables = self.hourly if variables is None else variables
        return {"latitude": lat, "longitude": lon,
                "start_date": start_date or self.start_date,
                "end_date": end_date or self.end_date,
                "hourly": ",".join(v + self.suffix for v in variables),
                "timezone": "GMT", "wind_speed_unit": "kn"}

    def check_variables(self, code):
        """Forecast mode: test each variable on two days at one airport, and
        drop any that the forecast archive rejects or returns empty. Not every
        observed-weather variable exists as a past forecast."""
        r = self.coords.loc[code]
        end = (pd.Timestamp(self.start_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Checking variables at {code} for a {self.lead_days}-day-ahead forecast:")
        keep = []
        for var in self.hourly:
            resp = requests.get(self.url, timeout=60, params=self.build_params(
                r["LATITUDE"], r["LONGITUDE"], [var], self.start_date, end))
            if resp.status_code == 429:
                raise RuntimeError("Rate limited during the variable check; try again later.")
            if not resp.ok:
                status = "rejected (" + error_text(resp) + ")"
            else:
                values = resp.json()["hourly"].get(var + self.suffix) or []
                status = "ok" if any(v is not None for v in values) else "no data"
            print(f"  {var}: {status}")
            if status == "ok":
                keep.append(var)
            time.sleep(PAUSE)
        if not keep:
            raise RuntimeError("No variable returned data. Read the messages above; "
                               "the request itself is probably being rejected.")
        dropped = [v for v in self.hourly if v not in keep]
        if dropped:
            print("Continuing WITHOUT:", dropped)
        self.hourly = keep

    def fetch_airport(self, code):
        out_file = self.out_dir / f"{code}.parquet"
        if out_file.exists():
            return "skipped"
        r = self.coords.loc[code]
        if r["tz"] is None:
            self.failed.append((code, "no time zone found"))
            return "failed"
        try:
            resp = requests.get(self.url, params=self.build_params(r["LATITUDE"], r["LONGITUDE"]),
                                timeout=60)
            if resp.status_code == 429:
                self.failed.append((code, "429 rate limit"))
                time.sleep(60)
                return "rate_limited"
            if not resp.ok:
                raise RuntimeError(error_text(resp))
            w = pd.DataFrame(resp.json()["hourly"])
            # Strip the lead-time suffix so the columns keep the usual names
            # (origin_precipitation, ...) whichever kind of weather was pulled.
            if self.suffix:
                w.columns = [c[:-len(self.suffix)] if c.endswith(self.suffix) else c
                             for c in w.columns]
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
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
    # If the file already has weather merged in (flights_data.csv), remove it
    # so the new weather replaces it instead of clashing with it.
    old_weather = [c for c in flights.columns if c.startswith(("origin_", "dest_"))]
    if old_weather:
        print(f"Replacing {len(old_weather)} weather columns already in {path}")
        flights = flights.drop(columns=old_weather)
    for col in ["CRS_DEP_TIME", "CRS_ARR_TIME"]:
        # Works whether the column was read as int, float or string
        flights[col] = pd.to_numeric(flights[col]).astype(int).astype(str).str.zfill(4)
    try:
        flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"])   # already a plain date
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
    print("Weather:", f"forecast made {LEAD_DAYS} day(s) ahead" if LEAD_DAYS else "observed")

    flights = add_datetimes(load_flights(DATA_FILE))
    print(f"{len(flights):,} flights after filtering "
          f"({int(flights['CANCELLED'].sum()):,} cancelled, {int(flights['DIVERTED'].sum()):,} diverted)")

    # Date range from the data; +1 day so overnight arrivals on the last day get weather
    start = flights["FL_DATE"].min().strftime("%Y-%m-%d")
    end = (flights["FL_DATE"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    api = FlightAPI(CODES_FILE, CACHE_DIR, start, end, hourly, LEAD_DAYS)

    codes = pd.concat([flights["ORIGIN"], flights["DEST"]]).unique()
    missing = [c for c in codes if c not in api.coords.index]
    print(f"{len(codes)} airports | missing from codes file: {missing}")
    known = [c for c in codes if c in api.coords.index]

    if LEAD_DAYS:
        # Test on the busiest airport before pulling all of them
        busiest = flights.loc[flights["ORIGIN"].isin(known), "ORIGIN"].mode()[0]
        api.check_variables(busiest)

    api.fetch_all(known)

    merged = merge_weather(flights, api.load())
    nan_rate = merged.filter(regex="^(origin|dest)_").isna().mean()
    print(f"Max missing-weather rate across columns: {nan_rate.max():.4%}")
    if nan_rate.max() > 0.02:
        print("Missing rate by column:")
        print((nan_rate[nan_rate > 0] * 100).round(2).sort_values(ascending=False).to_string())

    if OUTPUT_FILE.endswith(".csv"):
        merged.to_csv(OUTPUT_FILE, index=False)
    else:
        merged.to_parquet(OUTPUT_FILE, index=False)
    print("Saved", OUTPUT_FILE)