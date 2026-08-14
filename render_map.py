"""Render the publication map with Forge3D's hybrid terrain reference path."""

import json
import math
import os
from pathlib import Path

os.environ.setdefault("WGPU_BACKEND", "vulkan")

import numpy as np
from forge3d.path_tracing import hybrid_render_terrain_reference
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
WIDTH, HEIGHT = 1500, 950
WEST, SOUTH, EAST, NORTH = -79.63926826, 43.500, -79.11524635, 43.85546581
SPAN = 120.0
COLORS = np.array(
    [[102, 121, 135], [169, 179, 178], [217, 213, 189], [214, 165, 142], [187, 91, 81], [153, 55, 49]],
    np.float32,
) / 255
STOPS = np.array([0, 25, 50, 70, 90, 100], np.float32) / 100


def render_shade(height, city):
    lines, columns = height.shape
    depth_span = SPAN * (NORTH - SOUTH) * math.cos(math.radians(43.72)) / (EAST - WEST)
    camera = {
        "origin": (0.0, 1200.0, 0.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 0.0, -1.0),
        "fov_y": 5.927,
        "exposure": 1.0,
    }
    result = hybrid_render_terrain_reference(
        height,
        WIDTH,
        HEIGHT,
        camera,
        spacing=(SPAN / (columns - 1), depth_span / (lines - 1)),
        exaggeration=42.0,
        albedo=(0.68, 0.63, 0.58),
        sun_azimuth_deg=305,
        sun_elevation_deg=32,
        sun_intensity=1.40,
        env_intensity=0.96,
        spp=2,
        min_frames=96,
        max_frames=640,
        variance_threshold=0.004,
        seed=31,
    )
    assert result["converged"]

    depth = np.asarray(result["depth"], np.float32)
    hit = np.isfinite(depth)
    origin = np.array(camera["origin"], np.float32)
    forward = np.array(camera["look_at"], np.float32) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array(camera["up"], np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    py, px = np.mgrid[0:HEIGHT, 0:WIDTH]
    tangent = math.tan(math.radians(camera["fov_y"]) / 2)
    ray = forward + (2 * (px + 0.5) / WIDTH - 1)[..., None] * tangent * (WIDTH / HEIGHT) * right
    ray += (1 - 2 * (py + 0.5) / HEIGHT)[..., None] * tangent * up
    ray /= np.linalg.norm(ray, axis=2, keepdims=True)
    position = origin + ray * depth[..., None]
    i = (np.nan_to_num(np.clip(position[..., 0] / SPAN + 0.5, 0, 1)) * (columns - 1)).astype(int)
    j = ((1 - np.nan_to_num(np.clip(position[..., 2] / depth_span + 0.5, 0, 1))) * (lines - 1)).astype(int)
    normal = np.asarray(result["normal"], np.float32)
    rgba = np.asarray(result["rgba"], np.float32)[..., :3] / 255
    top = hit & (normal[..., 1] > 0.2)
    accumulation = np.zeros((lines, columns, 3), np.float32)
    count = np.zeros((lines, columns), np.float32)
    np.add.at(accumulation, (j[top], i[top]), rgba[top])
    np.add.at(count, (j[top], i[top]), 1)
    missing = count == 0
    if missing.any():
        _, nearest = distance_transform_edt(missing, return_indices=True)
        accumulation = accumulation[nearest[0], nearest[1]]
        count = np.maximum(count[nearest[0], nearest[1]], 1)
    shade = accumulation / count[..., None]

    z = np.flipud(height) * 42.0
    gy, gx = np.gradient(z, depth_span / (lines - 1), SPAN / (columns - 1))
    azimuth, elevation = math.radians(305), math.radians(26)
    lx, ly, lz = math.cos(elevation) * math.sin(azimuth), math.sin(elevation), math.cos(elevation) * math.cos(azimuth)
    hillshade = np.clip((-gx * lx + ly - gy * lz) / (np.sqrt(gx * gx + gy * gy + 1) + 1e-6), 0, 1)
    shade = np.clip(shade * (0.38 + 0.85 * hillshade[..., None]), 0.52, 1.55)
    shade = np.clip(shade / (np.mean(shade[city], axis=0) + 1e-6), 0.58, 1.52)
    return shade, {"frames": result["frames"], "variance": result["variance"]}


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
    ]
    return next((ImageFont.truetype(path, size) for path in candidates if path.exists()), ImageFont.load_default(size=size))


def compose(texture, city, shade):
    lit = np.clip(texture * gaussian_filter(shade, (3.2, 3.2, 0)), 0, 1)
    canvas = np.ones((*city.shape, 3), np.float32)
    canvas[city] = lit[city]
    image = ImageEnhance.Contrast(
        ImageEnhance.Color(Image.fromarray((canvas * 255).astype(np.uint8))).enhance(1.12)
    ).enhance(1.04)

    pixels = np.asarray(image)
    subject = np.any(np.abs(pixels - 255) > 20, axis=2)
    yy, xx = np.where(subject)
    x0, x1 = max(0, xx.min() - 30), min(image.width, xx.max() + 31)
    y0, y1 = max(0, yy.min() - 24), min(image.height, yy.max() + 25)
    crop = image.crop((x0, y0, x1, y1))
    width, height_px = crop.size
    target = 1500 / 950
    pad_x = pad_y = 0
    if width / height_px > target:
        padded_height = round(width / target)
        pad_y = (padded_height - height_px) // 2
        padded = Image.new("RGB", (width, padded_height), "white")
        padded.paste(crop, (0, pad_y))
    else:
        padded_width = round(height_px * target)
        pad_x = (padded_width - width) // 2
        padded = Image.new("RGB", (padded_width, height_px), "white")
        padded.paste(crop, (pad_x, 0))
    source_width, source_height = padded.size
    image = padded.resize((1600, 1013), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    place_font = font(20, bold=True)
    for name, lon, lat, dx, dy in [
        ("Etobicoke", -79.5435, 43.6537, -92, -31),
        ("Downtown", -79.3832, 43.6532, 12, -31),
        ("North York", -79.4111, 43.7615, 12, -31),
        ("Scarborough", -79.2318, 43.7764, -118, -31),
    ]:
        px = (((lon - WEST) / (EAST - WEST) * (city.shape[1] - 1) - x0) + pad_x) * 1600 / source_width
        py = (((NORTH - lat) / (NORTH - SOUTH) * (city.shape[0] - 1) - y0) + pad_y) * 1013 / source_height
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(23, 24, 21))
        draw.text((px + dx, py + dy), name, font=place_font, fill=(23, 24, 21), stroke_width=3, stroke_fill="white")

    title_font = font(38, bold=True)
    label_font = font(20)
    legend_font = font(18, bold=True)
    x, y = 1600 - 46 - 400, 1013 - 190
    ink = (30, 34, 42)
    draw.text((x + 400, y), "Toronto Walk Score", font=title_font, fill=ink, anchor="ra")
    draw.text((x + 400, y + 68), "WALK SCORE", font=legend_font, fill=(72, 78, 87), anchor="ra")
    bar_y = y + 100
    for column in range(400):
        value = column / 399
        colour = tuple(int(np.interp(value, STOPS, COLORS[:, channel]) * 255) for channel in range(3))
        draw.line((x + column, bar_y, x + column, bar_y + 18), fill=colour)
    draw.text((x, bar_y + 22), "LOW / FLAT", font=label_font, fill=ink)
    draw.text((x + 400, bar_y + 22), "HIGH / STEEP", font=label_font, fill=ink, anchor="ra")
    return image


def main():
    OUTPUT.mkdir(exist_ok=True)
    inputs = np.load(ROOT / "toronto_inputs.npz")
    height = inputs["height"]
    texture = inputs["texture"].astype(np.float32) / 255
    city = inputs["city"]
    shade, metadata = render_shade(height, city)
    image = compose(texture, city, shade)
    image.save(OUTPUT / "toronto-walk-score-forge3d.png")
    (OUTPUT / "render-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
