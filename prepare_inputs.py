"""Build the heightfield, city mask, and UV texture from the archived source data."""

import csv
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import shapefile
from PIL import Image, ImageDraw
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, grey_closing

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BOUNDS = (-79.63926826, 43.500, -79.11524635, 43.85546581)
GRID = (1600, 934)
STOPS = np.array([0, 25, 50, 70, 90, 100], np.float32) / 100
COLORS = np.array(
    [[102, 121, 135], [169, 179, 178], [217, 213, 189], [214, 165, 142], [187, 91, 81], [153, 55, 49]],
    np.float32,
) / 255


def world(lon, lat, zoom):
    scale = 2**zoom * 256
    return (
        (lon + 180) / 360 * scale,
        (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * scale,
    )


def fetch_basemap(bounds, zoom=13):
    west, south, east, north = bounds
    x0, y0 = world(west, north, zoom)
    x1, y1 = world(east, south, zoom)
    tx0, ty0, tx1, ty1 = map(int, (x0 // 256, y0 // 256, x1 // 256, y1 // 256))
    tile_size = 512
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * tile_size, (ty1 - ty0 + 1) * tile_size))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            request = urllib.request.Request(
                f"https://a.basemaps.cartocdn.com/light_nolabels/{zoom}/{tx}/{ty}@2x.png",
                headers={"User-Agent": "Toronto Walk Score Forge3D/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                tile = Image.open(io.BytesIO(response.read())).convert("RGB")
            mosaic.paste(tile, ((tx - tx0) * tile_size, (ty - ty0) * tile_size))
    crop = tuple(round(value * 2) for value in (x0 - tx0 * 256, y0 - ty0 * 256, x1 - tx0 * 256, y1 - ty0 * 256))
    return mosaic.crop(crop)


def score_surface(rows, ring):
    west, south, east, north = BOUNDS
    columns, lines = GRID
    xs, ys = np.linspace(west, east, columns), np.linspace(south, north, lines)
    xx, yy = np.meshgrid(xs, ys)
    points = np.array([[float(row["lon"]), float(row["lat"])] for row in rows])
    scores = np.array([float(row["score"]) for row in rows])
    cubic = griddata(points, scores, (xx, yy), method="cubic")
    nearest = griddata(points, scores, (xx, yy), method="nearest")
    raw = np.where(np.isnan(cubic), nearest, cubic)
    broad = gaussian_filter(raw, (30, 46))
    medium = gaussian_filter(raw, (14, 22))
    fine = gaussian_filter(raw, (5, 8))
    surface = 0.22 * broad + 0.38 * medium + 0.40 * fine + 0.10 * (fine - medium)

    mask = Image.new("L", GRID)
    ImageDraw.Draw(mask).polygon(
        [((lon - west) / (east - west) * (columns - 1), (north - lat) / (north - south) * (lines - 1)) for lon, lat in ring],
        fill=255,
    )
    city = np.asarray(mask) > 0
    surface = np.clip((surface - 12) / (99 - 12), 0, 1)
    gy, gx = np.gradient(surface)
    ridge = gaussian_filter(np.hypot(gx, gy), 6)
    ridge /= np.percentile(ridge[city], 98) + 1e-6
    surface = np.clip(surface + 0.11 * np.clip(ridge, 0, 1) * (0.35 + 0.65 * surface), 0, 1)
    return (0.03 + 0.97 * surface**1.10).astype(np.float32), np.flipud(surface), city


def map_texture(level, basemap):
    columns, lines = GRID
    score_rgb = np.stack([np.interp(level, STOPS, COLORS[:, channel]) for channel in range(3)], axis=2)
    grey = np.asarray(basemap.convert("L"), np.float32) / 255
    paper = grey_closing(grey, size=(7, 7))
    alpha = np.clip((paper - grey - 0.003) * 12, 0, 0.48)
    alpha = np.asarray(
        Image.fromarray((alpha * 255).astype(np.uint8)).resize(GRID, Image.Resampling.LANCZOS),
        np.float32,
    )[..., None] / 255
    ink = np.array([45, 48, 45], np.float32) / 255
    texture = score_rgb * (1 - alpha) + ink * alpha

    west, south, east, north = BOUNDS
    roads = Image.new("L", GRID)
    draw = ImageDraw.Draw(roads)
    elements = json.loads((DATA / "gta_major_roads.json").read_text(encoding="utf-8"))["elements"]
    for element in elements:
        if element.get("tags", {}).get("highway") not in ("motorway", "trunk"):
            continue
        points = [
            ((node["lon"] - west) / (east - west) * (columns - 1), (north - node["lat"]) / (north - south) * (lines - 1))
            for node in element.get("geometry", [])
        ]
        if len(points) > 1:
            draw.line(points, fill=105, width=1, joint="curve")
    vector_alpha = (np.asarray(roads, np.float32) / 255)[..., None]
    return texture * (1 - vector_alpha) + ink * vector_alpha


def main():
    rows = [row for row in csv.DictReader((DATA / "toronto_walkscore_extended.csv").open(encoding="utf-8")) if row["score"]]
    assert len(rows) == 796, f"expected archived 796 valid scores, found {len(rows)}"
    ring = shapefile.Reader(str(DATA / "boundary" / "citygcs_regional_mun_wgs84.shp")).shape(0).points
    height, level, city = score_surface(rows, ring)
    texture = map_texture(level, fetch_basemap(BOUNDS))
    np.savez_compressed(
        ROOT / "toronto_inputs.npz",
        height=height,
        texture=np.rint(texture * 255).astype(np.uint8),
        city=city,
    )
    print(f"wrote toronto_inputs.npz from {len(rows)} scores")


if __name__ == "__main__":
    main()
