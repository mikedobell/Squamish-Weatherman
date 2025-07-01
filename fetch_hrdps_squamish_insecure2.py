#!/usr/bin/env python3
"""
Print marine forecasts for Howe Sound, then wind and 2 m temperature for multiple locations,
with dual highlighting:
 - Bright green when temperatures strictly increase from Furry→Brit→7mesh→Cheak→Whis→Pemb→Lill.
 - Yellow when they increase from Furry→Brit→7mesh→Whis only.
Wind direction is shown as compass points (e.g. N, NNW, W).
If today's marine forecast contains certain keywords, highlight accordingly on the specific phrases:
  - "STRONG WIND WARNING IN EFFECT" → red only that phrase
  - "inflow...northern sections" → green only that phrase
  - "outflow...southern sections" → blue only that phrase
Exclude any RSS entries whose title starts with "Extended Forecast".
Only forecasts between 05:00 and 20:00 local time are shown, with a horizontal line after 20:00 each day.
Each day’s forecasts are preceded by a header like "Tuesday 03 June 2025".
"""

import argparse
import datetime
from zoneinfo import ZoneInfo
import tempfile
import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pygrib
import urllib3
import re

# Suppress insecure TLS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ANSI highlight codes (raw escapes)
H_RED    = "\033[91m"
H_GREEN  = "\033[92m"
H_YELLOW = "\033[93m"
H_BLUE   = "\033[94m"
H_RESET  = "\033[0m"

# 16-point compass conversion
COMPASS_POINTS = [
    'N','NNE','NE','ENE','E','ESE','SE','SSE',
    'S','SSW','SW','WSW','W','WNW','NW','NNW'
]

def deg_to_compass(deg: float) -> str:
    idx = int((deg + 11.25) / 22.5) % 16
    return COMPASS_POINTS[idx]

# Marine forecast functions
def get_marine_forecast(rss_url: str, region_filter: str):
    try:
        resp = requests.get(rss_url)
        resp.raise_for_status()
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(resp.content)
        forecasts = []
        for entry in root.findall('.//atom:entry', ns):
            title = entry.find('atom:title', ns).text or ''
            # skip extended forecasts
            if title.lower().startswith('extended forecast'):
                continue
            if region_filter in title:
                summary = entry.find('atom:summary', ns).text or ''
                published = entry.find('atom:published', ns).text or ''
                # Format publication
                try:
                    dt = datetime.datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    pub = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pub = published
                # first line before HTML break
                forecast_text = summary.split('<br/>')[0].strip()
                forecasts.append({'Title': title, 'Forecast': forecast_text, 'Published': pub})
        return forecasts
    except Exception as e:
        print(f"Error retrieving marine forecast: {e}")
        return []


def display_marine_forecasts():
    howe_rss = "https://weather.gc.ca/rss/marine/06400_e.xml"
    print("\nForecast for Today, Tonight and Sunday - Howe Sound")
    print("=" * 75)
    entries = get_marine_forecast(howe_rss, "Howe Sound")
    if not entries:
        print("No marine forecasts available for Howe Sound.\n")
        return
    for fc in entries:
        title = fc['Title']
        text = fc['Forecast']
        # highlight phrases only
        text = text.replace(
            "STRONG WIND WARNING IN EFFECT",
            f"{H_RED}STRONG WIND WARNING IN EFFECT{H_RESET}"
        )
        text = re.sub(
            r"(inflow.*?northern sections)",
            lambda m: f"{H_GREEN}{m.group(1)}{H_RESET}",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"(outflow.*?southern sections)",
            lambda m: f"{H_BLUE}{m.group(1)}{H_RESET}",
            text,
            flags=re.IGNORECASE
        )
        # print with highlights
        print(f"\n{title}")
        print("-" * 75)
        print(f"Forecast: {text}")
        print(f"Published: {fc['Published']}")

# Locations for wind & temperature
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
W_TIME, W_SPEED, W_DIR, W_TEMP = 10, 7, 6, 7


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
    return (
        f"{base}/CMC_hrdps_west_UGRD_TGL_10_{RESOLUTION}_{ds}_P{fh3}-00.grib2",
        f"{base}/CMC_hrdps_west_VGRD_TGL_10_{RESOLUTION}_{ds}_P{fh3}-00.grib2",
        f"{base}/CMC_hrdps_west_TMP_TGL_2_{RESOLUTION}_{ds}_P{fh3}-00.grib2",
    )


def download(url):
    r = requests.get(url, stream=True, verify=False, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
    for chunk in r.iter_content(8192): tmp.write(chunk)
    tmp.close(); return tmp.name


def extract_wind(ufile, vfile):
    fu = pygrib.open(ufile)
    mu = fu.select(name='10 metre U wind component')[0]
    lats, lons = mu.latlons()
    d2 = (lats - SQUAMISH_LAT)**2 + (lons - SQUAMISH_LON)**2
    iy, ix = np.unravel_index(d2.argmin(), d2.shape)
    u = mu.values[iy, ix]; fu.close()
    fv = pygrib.open(vfile)
    mv = fv.select(name='10 metre V wind component')[0]
    v = mv.values[iy, ix]; fv.close()
    return float(u), float(v)


def extract_temps(tfile):
    ft = pygrib.open(tfile)
    mt = ft.select(name='2 metre temperature')[0]
    lats, lons = mt.latlons(); vals = mt.values; ft.close()
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
    uurl, _, _ = grib_urls(run, 0)
    try:
        if requests.head(uurl, verify=False, timeout=5).status_code != 200:
            run -= datetime.timedelta(hours=12)
    except:
        run -= datetime.timedelta(hours=12)
    return run


def main():
    args = parse_args()
    display_marine_forecasts()
    if args.run:
        run_dt = datetime.datetime.strptime(args.run, "%Y-%m-%dT%HZ").replace(tzinfo=datetime.timezone.utc)
    else:
        run_dt = find_best_run()
    hours = range(args.start, args.end+1, args.step)

    # print table header
    hdr = f"{'Time':<{W_TIME}} {'Spd(kt)':>{W_SPEED}} {'Dir':>{W_DIR}}"
    for lbl,_,_ in LOCS: hdr += f" {lbl:>{W_TEMP}}"
    print(hdr)
    print("-" * len(hdr))

    last_date = None
    for fh in hours:
        valid_utc = run_dt + datetime.timedelta(hours=fh)
        loc = valid_utc.astimezone(LOCAL_TZ)
        hr = loc.hour
        if hr < 5 or hr > 20:
            continue
        if loc.date() != last_date:
            if last_date:
                print()
            print(loc.strftime("%A %d %B %Y"))
            last_date = loc.date()

        ts_plain = loc.strftime("%H:%M %Z")
        ts = ts_plain.ljust(W_TIME)

        # download & extract
        uurl, vurl, turl = grib_urls(run_dt, fh)
        uf, vf, tf = download(uurl), download(vurl), download(turl)
        try:
            u, v = extract_wind(uf, vf)
            temps = extract_temps(tf)
        finally:
            for tmpf in (uf, vf, tf): os.remove(tmpf)

        spd = np.hypot(u, v) * 1.94384
        dir_str = deg_to_compass(dir_met(u, v)).rjust(W_DIR)

        # highlight time field based on thermal sequence
        seq_full = [temps[lbl] for lbl,_,_ in LOCS]
        if all(seq_full[i] < seq_full[i+1] for i in range(len(seq_full)-1)):
            ts = f"{H_GREEN}{ts}{H_RESET}"
        else:
            part = [temps['Furry'], temps['Brit'], temps['7mesh'], temps['Whis']]
            if all(part[i] < part[i+1] for i in range(len(part)-1)):
                ts = f"{H_YELLOW}{ts}{H_RESET}"

        # build and print row
        row = f"{ts} {spd:>{W_SPEED}.1f} {dir_str}"
        for lbl,_,_ in LOCS:
            row += f" {temps[lbl]:>{W_TEMP}.1f}"
        print(row)

        # separator after 20:00
        if hr == 20:
            print("-" * len(hdr))

if __name__ == "__main__":
    main()
