#!/usr/bin/env python3
"""
Print wind and 2 m temperature forecasts for multiple locations,
highlighting times when the temperature strictly increases from
Furry→Brit→7mesh→Whis by wrapping the time in yellow.
"""

import argparse
import datetime
from zoneinfo import ZoneInfo
import tempfile
import os
import requests
import numpy as np
import pygrib
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ANSI highlight codes
H_YELLOW = "\033[93m"
H_RESET  = "\033[0m"

# Short labels and their coords, in thermal‐wind order:
LOCS = [
    ("Furry",  49.57399,   -123.25493),
    ("Brit",   49.62383,   -123.23021),
    ("7mesh",  49.70100,   -123.15500),
    ("Cheak",  49.78259,   -123.17802),
    ("Whis",   50.12938,   -122.96207),
    ("Pemb",   50.31971,   -122.80706),
    ("Lill",   50.69374,   -121.93417),
]

SQUAMISH_LAT, SQUAMISH_LON = 49.70100, -123.15500
LOCAL_TZ = ZoneInfo("America/Vancouver")
BASE_URL   = "https://dd.alpha.weather.gc.ca/model_hrdps/west/1km/grib2"
RESOLUTION = "rotated_latlon0.009x0.009"

# column widths
W_TIME  = 10   # e.g. "05:00 PDT"
W_SPEED = 7    # e.g. " 12.3"
W_DIR   = 6    # e.g. " 245"
W_TEMP  = 7    # e.g. " 12.7"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=48)
    p.add_argument("--step",  type=int, default=3)
    p.add_argument("--run",   type=str, default=None)
    return p.parse_args()

def grib_urls(run_dt, fh):
    hh = run_dt.strftime("%H")
    ds = run_dt.strftime("%Y%m%dT%HZ")
    fh3 = f"{fh:03d}"
    base = f"{BASE_URL}/{hh}/{fh3}"
    u = f"{base}/CMC_hrdps_west_UGRD_TGL_10_{RESOLUTION}_{ds}_P{fh3}-00.grib2"
    v = f"{base}/CMC_hrdps_west_VGRD_TGL_10_{RESOLUTION}_{ds}_P{fh3}-00.grib2"
    t = f"{base}/CMC_hrdps_west_TMP_TGL_2_{RESOLUTION}_{ds}_P{fh3}-00.grib2"
    return u, v, t

def download(url):
    r = requests.get(url, stream=True, verify=False, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
    for chunk in r.iter_content(8192):
        f.write(chunk)
    f.close()
    return f.name

def extract_wind(u_path, v_path):
    fu = pygrib.open(u_path)
    mu = fu.select(name='10 metre U wind component')[0]
    lats, lons = mu.latlons()
    d2 = (lats - SQUAMISH_LAT)**2 + (lons - SQUAMISH_LON)**2
    iy, ix = np.unravel_index(d2.argmin(), d2.shape)
    u = mu.values[iy, ix]
    fu.close()

    fv = pygrib.open(v_path)
    mv = fv.select(name='10 metre V wind component')[0]
    v = mv.values[iy, ix]
    fv.close()

    return float(u), float(v)

def extract_temps(t_path):
    ft = pygrib.open(t_path)
    mt = ft.select(name='2 metre temperature')[0]
    lats, lons = mt.latlons()
    vals = mt.values
    ft.close()
    temps = {}
    for label, lat0, lon0 in LOCS:
        d2 = (lats - lat0)**2 + (lons - lon0)**2
        iy, ix = np.unravel_index(d2.argmin(), d2.shape)
        temps[label] = float(vals[iy, ix] - 273.15)
    return temps

def dir_met(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360

def find_best_run():
    now = datetime.datetime.now(datetime.timezone.utc)
    h = 12 if now.hour >= 12 else 0
    run = now.replace(hour=h, minute=0, second=0, microsecond=0)
    u_url, _, _ = grib_urls(run, 0)
    try:
        if requests.head(u_url, verify=False, timeout=5).status_code != 200:
            run -= datetime.timedelta(hours=12)
    except:
        run -= datetime.timedelta(hours=12)
    return run

def main():
    args = parse_args()
    if args.run:
        run_dt = datetime.datetime.strptime(args.run, "%Y-%m-%dT%HZ")
        run_dt = run_dt.replace(tzinfo=datetime.timezone.utc)
    else:
        run_dt = find_best_run()

    hours = range(args.start, args.end+1, args.step)

    # header
    hdr = f"{'Time':<{W_TIME}} {'Spd(kt)':>{W_SPEED}} {'Dir':>{W_DIR}}"
    for label,_,_ in LOCS:
        hdr += f" {label:>{W_TEMP}}"
    print(hdr)
    print("-" * len(hdr))

    last_date = None
    for fh in hours:
        valid_utc = run_dt + datetime.timedelta(hours=fh)
        loc = valid_utc.astimezone(LOCAL_TZ)
        hr = loc.hour
        if hr < 5 or hr > 20:
            continue

        # new day header
        if loc.date() != last_date:
            if last_date is not None:
                print()
            print(loc.strftime("%A %d %B %Y"))
            last_date = loc.date()

        # prepare time string
        ts_plain = loc.strftime("%H:%M") + f" {loc.tzname()}"
        ts = ts_plain.ljust(W_TIME)

        try:
            uurl, vurl, turl = grib_urls(run_dt, fh)
            uf = download(uurl); vf = download(vurl); tf = download(turl)
            u, v = extract_wind(uf, vf)
            temps = extract_temps(tf)
        finally:
            for f in (uf, vf, tf):
                if os.path.exists(f): os.remove(f)

        # wind in knots and direction
        spd_kt = np.hypot(u, v) * 1.94384
        d = dir_met(u, v)

        # check strictly increasing for Furry→Brit→7mesh→Whis
        seq = [temps["Furry"], temps["Brit"], temps["7mesh"], temps["Whis"]]
        if seq[0] < seq[1] < seq[2] < seq[3]:
            ts = f"{H_YELLOW}{ts}{H_RESET}"

        # build row
        row = f"{ts} {spd_kt:>{W_SPEED}.1f} {d:>{W_DIR}.0f}"
        for label,_,_ in LOCS:
            row += f" {temps[label]:>{W_TEMP}.1f}"
        print(row)

        if hr == 20:
            print("-" * len(hdr))

if __name__ == "__main__":
    main()
