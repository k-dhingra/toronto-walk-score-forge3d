"""Render Toronto Walk Score as a colour-on-surface, oblique path-traced map.

Unlike the archival top-down composition, this renderer samples the Walk Score
texture at each path-traced terrain hit.  The colour therefore stays attached to
the corresponding heightfield point when the camera is tilted.
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("WGPU_BACKEND", "vulkan")

import numpy as np
from forge3d.path_tracing import hybrid_render_terrain_reference
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
WEST, SOUTH, EAST, NORTH = -79.63926826, 43.500, -79.11524635, 43.85546581
SPAN = 120.0
# Restrained relief: this remains visibly oblique without turning the score
# field and the city boundary into a mountain range.
EXAGGERATION = 10.0
CAMERA_AZIMUTH = 180.0  # due south: west remains left, east remains right, north remains up
CAMERA_ELEVATION = 60.0
CAMERA_FOV_Y = 20.0
# The terrain fills the publication canvas; furniture is overlaid translucently.
# Native frame buffers are limited to the measured Vulkan budget, while the
# composition still publishes at the full poster size.
RENDER_SIZE = (1120, 709)
POSTER_SIZE = (3840, 2431)
POSTER_HEIGHT = POSTER_SIZE[1]
TERRAIN_RESOLUTION_SCALE = 2.56
COLORS = np.array(
    [[102, 121, 135], [169, 179, 178], [217, 213, 189], [214, 165, 142], [187, 91, 81], [153, 55, 49]],
    np.float32,
) / 255
STOPS = np.array([0, 25, 50, 70, 90, 100], np.float32) / 100


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
    ]
    return next((ImageFont.truetype(path, size) for path in candidates if path.exists()), ImageFont.load_default(size=size))


def field_dimensions(height):
    lines, columns = height.shape
    # X is longitude scaled at this latitude; divide by cos(latitude) to retain
    # the local metre aspect ratio for the north/south latitude span.
    depth_span = SPAN * (NORTH - SOUTH) / (math.cos(math.radians(43.72)) * (EAST - WEST))
    return lines, columns, depth_span


def increase_terrain_resolution(height, texture, city):
    """Increase mesh tessellation without inventing new score observations."""
    lines, columns = height.shape
    target = (round(columns * TERRAIN_RESOLUTION_SCALE), round(lines * TERRAIN_RESOLUTION_SCALE))
    terrain = np.clip(
        np.asarray(Image.fromarray(height, mode="F").resize(target, Image.Resampling.BICUBIC), np.float32),
        float(height.min()),
        float(height.max()),
    )
    texture_rgb = np.asarray(
        Image.fromarray(np.rint(texture * 255).astype(np.uint8)).resize(target, Image.Resampling.LANCZOS),
        np.float32,
    ) / 255
    mask = np.asarray(
        Image.fromarray(city.astype(np.uint8) * 255).resize(target, Image.Resampling.NEAREST),
        np.uint8,
    ) > 0
    return terrain, texture_rgb, mask


def camera_for_field(height, depth_span):
    """Frame the complete field from due south, with all distances in map units."""
    relief = EXAGGERATION * float(height.max() - height.min())
    distance = 1.75 * math.hypot(SPAN, depth_span)
    elevation = math.radians(CAMERA_ELEVATION)
    azimuth = math.radians(CAMERA_AZIMUTH)
    ground_distance = distance * math.cos(elevation)
    return {
        "origin": (
            ground_distance * math.sin(azimuth),
            distance * math.sin(elevation),
            ground_distance * math.cos(azimuth),
        ),
        "look_at": (0.0, 0.22 * relief, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": CAMERA_FOV_Y,
        "exposure": 1.0,
    }


def camera_basis(camera):
    origin = np.asarray(camera["origin"], np.float32)
    forward = np.asarray(camera["look_at"], np.float32) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera["up"], np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return origin, forward, right, up


def primary_rays(camera, width, height):
    origin, forward, right, up = camera_basis(camera)
    py, px = np.mgrid[0:height, 0:width]
    tangent = math.tan(math.radians(camera["fov_y"]) / 2)
    rays = forward + (2 * (px + 0.5) / width - 1)[..., None] * tangent * (width / height) * right
    rays += (1 - 2 * (py + 0.5) / height)[..., None] * tangent * up
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    return origin, forward, right, up, rays


def project_points(points, camera, width, height):
    """Project world points with the same pinhole camera used by the renderer."""
    origin, forward, right, up = camera_basis(camera)
    relative = points - origin
    depth = relative @ forward
    tangent = math.tan(math.radians(camera["fov_y"]) / 2)
    ndc_x = (relative @ right) / (depth * tangent * (width / height))
    ndc_y = (relative @ up) / (depth * tangent)
    return np.column_stack(((ndc_x + 1) * width / 2, (1 - ndc_y) * height / 2))


def sample_world_texture(texture, city, height, camera, width, height_px, depth_span):
    """Return Walk Score colours at terrain hits and the evidence for their coupling."""
    result = hybrid_render_terrain_reference(
        height,
        width,
        height_px,
        camera,
        spacing=(SPAN / (height.shape[1] - 1), depth_span / (height.shape[0] - 1)),
        exaggeration=EXAGGERATION,
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
    if not result["converged"]:
        raise RuntimeError("The path tracer did not converge.")

    lines, columns = height.shape
    origin, _, _, _, rays = primary_rays(camera, width, height_px)
    depth = np.asarray(result["depth"], np.float32)
    hit = np.isfinite(depth)
    position = origin + rays * np.where(hit, depth, 0)[..., None]
    column = np.rint(np.clip(position[..., 0] / SPAN + 0.5, 0, 1) * (columns - 1)).astype(int)
    raw_row = np.rint(np.clip(position[..., 2] / depth_span + 0.5, 0, 1) * (lines - 1)).astype(int)
    texture_row = lines - 1 - raw_row  # source texture is north-up; native height is south-up.
    visible = hit & city[texture_row, column]

    traced = np.asarray(result["rgba"], np.float32)[..., :3] / 255
    luma = traced @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    reference_luma = float(np.percentile(luma[visible], 92))
    lighting = np.clip(luma / max(reference_luma, 1e-6), 0.42, 1.08)
    rgb = np.ones((height_px, width, 3), np.float32)
    rgb[visible] = np.clip(texture[texture_row[visible], column[visible]] * lighting[visible, None], 0, 1)

    level = np.clip(((np.flipud(height) - 0.03) / 0.97) ** (1 / 1.10), 0, 1)
    sampled_level = level[texture_row[visible], column[visible]]
    sampled_elevation = height[raw_row[visible], column[visible]]
    expected_elevation = 0.03 + 0.97 * sampled_level**1.10
    points = position[visible]
    flat_points = points.copy()
    flat_points[:, 1] = 0
    screen_y = project_points(points, camera, width, height_px)[:, 1]
    flat_screen_y = project_points(flat_points, camera, width, height_px)[:, 1]
    metrics = {
        "frames": int(result["frames"]),
        "variance": float(result["variance"]),
        "visible_path_traced_hits": int(visible.sum()),
        "surface_score_height_max_abs_error": float(np.max(np.abs(sampled_elevation - expected_elevation))),
        "surface_score_height_correlation": float(np.corrcoef(sampled_level, sampled_elevation)[0, 1]),
        "screen_lift_from_geometry_px": {
            "min": float(np.min(flat_screen_y - screen_y)),
            "median": float(np.median(flat_screen_y - screen_y)),
            "max": float(np.max(flat_screen_y - screen_y)),
        },
    }
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8)), metrics, visible


def label_world_point(lon, lat, height, depth_span):
    lines, columns = height.shape
    column = int(round((lon - WEST) / (EAST - WEST) * (columns - 1)))
    raw_row = int(round((lat - SOUTH) / (NORTH - SOUTH) * (lines - 1)))
    return np.array(
        [
            (column / (columns - 1) - 0.5) * SPAN,
            float(height[raw_row, column]) * EXAGGERATION,
            (raw_row / (lines - 1) - 0.5) * depth_span,
        ],
        np.float32,
    )


def focus_surface(image, terrain_mask):
    """Frame all actual terrain hits, including pale southwest boundary pixels."""
    yy, xx = np.where(terrain_mask)
    if not len(xx):
        raise RuntimeError("The oblique renderer produced no visible terrain pixels.")
    x0, x1 = max(0, int(xx.min()) - 28), min(image.width, int(xx.max()) + 29)
    y0, y1 = max(0, int(yy.min()) - 22), min(image.height, int(yy.max()) + 23)
    crop = image.crop((x0, y0, x1, y1))
    crop_width, crop_height = crop.size
    target_aspect = POSTER_SIZE[0] / POSTER_SIZE[1]
    pad_x = pad_y = 0
    if crop_width / crop_height > target_aspect:
        padded_height = round(crop_width / target_aspect)
        pad_y = (padded_height - crop_height) // 2
        padded = Image.new("RGB", (crop_width, padded_height), "white")
        padded.paste(crop, (0, pad_y))
    else:
        padded_width = round(crop_height * target_aspect)
        pad_x = (padded_width - crop_width) // 2
        padded = Image.new("RGB", (padded_width, crop_height), "white")
        padded.paste(crop, (pad_x, 0))
    scale_x = POSTER_SIZE[0] / padded.width
    scale_y = POSTER_SIZE[1] / padded.height

    def transform(points):
        return np.column_stack(((points[:, 0] - x0 + pad_x) * scale_x, (points[:, 1] - y0 + pad_y) * scale_y))

    return padded.resize(POSTER_SIZE, Image.Resampling.LANCZOS), transform, {
        "source_crop": [x0, y0, x1, y1],
        "source_padding": [pad_x, pad_y],
    }


def decorate(image, projected_labels):
    """Add labels and a compact legend without changing the rendered geometry."""
    surface = image.convert("RGBA")
    width, surface_height = surface.size
    scale = width / 1600
    scaled = lambda value: round(value * scale)
    image = Image.new("RGBA", (width, POSTER_HEIGHT), "white")
    image.alpha_composite(surface, (0, 0))
    draw = ImageDraw.Draw(image)
    height_px = image.height
    place_font = font(scaled(20), bold=True)
    labels = [
        ("Etobicoke", -102, -32),
        ("Downtown", 12, -31),
        ("North York", 12, -31),
        ("Scarborough", -118, -31),
    ]
    for (name, dx, dy), (x, y) in zip(labels, projected_labels):
        if 0 <= x < width and 0 <= y < surface_height:
            draw.ellipse((x - scaled(4), y - scaled(4), x + scaled(4), y + scaled(4)), fill=(23, 24, 21, 255))
            draw.text((x + scaled(dx), y + scaled(dy)), name, font=place_font, fill=(23, 24, 21, 255), stroke_width=scaled(3), stroke_fill=(255, 255, 255, 230))

    # The lower-left terrain contains the southwest city edge.  Keep the
    # publication furniture in the empty upper-left water/negative space so
    # it never obscures Etobicoke or the rest of the map surface.
    panel_x, panel_y = scaled(28), scaled(26)
    panel = (panel_x, panel_y, scaled(558), panel_y + scaled(140))
    draw.rounded_rectangle(panel, radius=scaled(12), fill=(255, 255, 255, 232))
    ink = (30, 34, 42, 255)
    legend_x = panel_x + scaled(20)
    draw.text((legend_x, panel_y + scaled(17)), "Toronto Walk Score", font=font(scaled(34), bold=True), fill=ink)
    draw.text((legend_x, panel_y + scaled(61)), "WALK SCORE  •  HEIGHTFIELD", font=font(scaled(16), bold=True), fill=(72, 78, 87, 255))
    bar_x, bar_y, bar_width = legend_x, panel_y + scaled(91), scaled(380)
    for x in range(bar_width):
        value = x / (bar_width - 1)
        colour = tuple(int(np.interp(value, STOPS, COLORS[:, channel]) * 255) for channel in range(3))
        draw.line((bar_x + x, bar_y, bar_x + x, bar_y + scaled(17)), fill=colour + (255,))
    draw.text((bar_x, bar_y + scaled(23)), "LOW / FLAT", font=font(scaled(15)), fill=ink)
    draw.text((bar_x + bar_width, bar_y + scaled(23)), "HIGH / STEEP", font=font(scaled(15)), fill=ink, anchor="ra")
    draw.text(
        (width - scaled(28), height_px - scaled(25)),
        "Walk Score® data (independent) · © OpenStreetMap contributors · CARTO",
        font=font(scaled(12)),
        fill=(55, 59, 64, 255),
        anchor="rd",
    )
    return image.convert("RGB")


def main():
    global EXAGGERATION, CAMERA_AZIMUTH, CAMERA_ELEVATION, TERRAIN_RESOLUTION_SCALE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Render the same scene at half output resolution for framing inspection.")
    parser.add_argument("--output", type=Path, help="Output PNG path; defaults to the fixed publication image.")
    parser.add_argument("--exaggeration", type=float, default=EXAGGERATION, help="Vertical score-field scale in map units.")
    parser.add_argument("--camera-azimuth", type=float, default=CAMERA_AZIMUTH, help="Camera compass azimuth; 180 keeps the original north-up reading.")
    parser.add_argument("--camera-elevation", type=float, default=CAMERA_ELEVATION, help="Camera elevation above the ground plane in degrees.")
    parser.add_argument("--terrain-resolution-scale", type=float, default=TERRAIN_RESOLUTION_SCALE, help="Tessellation multiplier for the archived terrain grid.")
    args = parser.parse_args()
    if args.exaggeration <= 0:
        raise ValueError("--exaggeration must be positive.")
    if not 0 < args.camera_elevation < 90:
        raise ValueError("--camera-elevation must be between 0 and 90 degrees.")
    if args.terrain_resolution_scale < 1:
        raise ValueError("--terrain-resolution-scale must be at least 1.")
    EXAGGERATION = args.exaggeration
    CAMERA_AZIMUTH = args.camera_azimuth
    CAMERA_ELEVATION = args.camera_elevation
    TERRAIN_RESOLUTION_SCALE = args.terrain_resolution_scale

    render_width, render_height = RENDER_SIZE
    if args.probe:
        render_width //= 2
        render_height //= 2
    output_path = args.output or OUTPUT / ("toronto-walk-score-elevation-probe.png" if args.probe else "toronto-walk-score-elevation-fixed.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = np.load(ROOT / "toronto_inputs.npz")
    height, texture, city = increase_terrain_resolution(
        inputs["height"].astype(np.float32),
        inputs["texture"].astype(np.float32) / 255,
        inputs["city"].astype(bool),
    )
    _, _, depth_span = field_dimensions(height)
    camera = camera_for_field(height, depth_span)
    image, metrics, terrain_mask = sample_world_texture(texture, city, height, camera, render_width, render_height, depth_span)
    # A due-south oblique camera is horizontally mirrored relative to the
    # conventional north-up map register.  Mirror the finished camera raster
    # (and its annotations below) so west stays left and east stays right.
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    terrain_mask = np.fliplr(terrain_mask)
    labels = [
        (-79.5435, 43.6537),
        (-79.3832, 43.6532),
        (-79.4111, 43.7615),
        (-79.2318, 43.7764),
    ]
    world_labels = np.stack([label_world_point(lon, lat, height, depth_span) for lon, lat in labels])
    raw_label_positions = project_points(world_labels, camera, render_width, render_height)
    raw_label_positions[:, 0] = render_width - raw_label_positions[:, 0]
    image, transform_labels, focus_metadata = focus_surface(image, terrain_mask)
    image = decorate(image, transform_labels(raw_label_positions))
    image.save(output_path)
    metrics.update(
        {
            "output": str(output_path),
            "rendered_surface_size": [render_width, render_height],
            "poster_size": list(POSTER_SIZE),
            "camera": camera,
            "height_storage": "native rows south-to-north; texture rows north-to-south",
            "colour_binding": "Walk Score texture sampled at each path-traced terrain hit",
            "orientation": "north-up map register: due-south camera raster mirrored horizontally so west is left and east is right",
            "terrain_mesh": {
                "source_grid": [1600, 934],
                "tessellated_grid": [height.shape[1], height.shape[0]],
                "interpolation": "bicubic height / Lanczos texture / nearest city boundary; no new score observations",
            },
            "poster_framing": focus_metadata,
        }
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
