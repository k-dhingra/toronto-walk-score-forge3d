"""Refresh the archived Walk Score grid used by the map."""

import csv
import html
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BOUNDARY = DATA / "boundary" / "citygcs_regional_mun_wgs84.shp"
OUTPUT = DATA / "toronto_walkscore_extended.csv"
URL = "https://www.walkscore.com/score/loc/lat={lat:.6f}/lng={lon:.6f}"


def fetch(point):
    lat, lon = point
    url = URL.format(lat=lat, lon=lon)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Toronto Walk Score field study)"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            text = response.read().decode("utf-8", "ignore")
        match = re.search(r'alt="(\d{1,3}) Walk Score of [^"]+"', text) or re.search(
            r"has a Walk Score of (\d{1,3}) out of 100", text
        )
        place = re.search(r"<title>(.*?) - Walk Score</title>", text, re.S)
        return {
            "lat": lat,
            "lon": lon,
            "score": int(match.group(1)) if match else "",
            "place": html.unescape(place.group(1).strip()) if place else "",
            "url": url,
        }
    except Exception:
        return {"lat": lat, "lon": lon, "score": "", "place": "", "url": url}


def inside(lon, lat, ring):
    hit = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def points():
    ring = shapefile.Reader(str(BOUNDARY)).shape(0).points
    west, east = min(x for x, _ in ring), max(x for x, _ in ring)
    south, north = min(y for _, y in ring), max(y for _, y in ring)
    result = set()
    lat = south
    while lat <= north:
        lon = west
        while lon <= east:
            if inside(lon, lat, ring):
                result.add((round(lat, 8), round(lon, 8)))
            lon += 0.022
        lat += 0.022
    lat = 43.30
    while lat <= 44.10:
        lon = -79.90
        while lon <= -78.90:
            result.add((round(lat, 8), round(lon, 8)))
            lon += 0.035
        lat += 0.035
    return sorted(result, key=lambda point: (-point[0], point[1]))


def main():
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = [future.result() for future in as_completed(pool.submit(fetch, point) for point in points())]
    rows = sorted((row for row in rows if row["score"] != ""), key=lambda row: (-row["lat"], row["lon"]))
    if len(rows) < 500:
        raise SystemExit(f"only {len(rows)} valid scores; refusing to replace the archive")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["lat", "lon", "score", "place", "url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} valid scores to {OUTPUT}")


if __name__ == "__main__":
    main()
