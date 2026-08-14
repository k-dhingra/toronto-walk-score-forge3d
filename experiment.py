"""Fixed-sample Forge3D experiment: 64 raw, 264 raw, then A-Trous."""

import json
import math
import os
from pathlib import Path

os.environ.setdefault("WGPU_BACKEND", "vulkan")

import forge3d as f3d
import numpy as np
from forge3d import offline
from forge3d.denoise import atrous_denoise
from forge3d.terrain_params import DenoiseSettings, PomSettings, make_terrain_params_config
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SIZE = (1500, 950)
SPAN = 120.0
SEED = 31
CAMERA = {"radius": 235.0, "phi": 270.0, "theta": 30.0, "fov": 36.0}
COLORS = ["#667987", "#a9b3b2", "#d9d5bd", "#d6a58e", "#bb5b51", "#993731"]
STOPS = np.array([0.0, 0.25, 0.5, 0.7, 0.9, 1.0])


def write_environment(path):
    if path.exists():
        return
    with path.open("wb") as stream:
        stream.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 16 +X 32\n")
        for y in range(16):
            for x in range(32):
                stream.write(bytes((190 + min(x, 31), 196 + min(y, 15), 210, 129)))


def params(samples):
    overlay = f3d.OverlayLayer.from_colormap1d(
        f3d.Colormap1D.from_stops(list(zip(STOPS, COLORS)), domain=(0.0, 1.0)),
        strength=1.0,
    )
    config = make_terrain_params_config(
        size_px=SIZE,
        render_scale=1.0,
        terrain_span=SPAN,
        msaa_samples=1,
        z_scale=14.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        light_azimuth_deg=305.0,
        light_elevation_deg=32.0,
        sun_intensity=1.4,
        cam_radius=CAMERA["radius"],
        cam_phi_deg=CAMERA["phi"],
        cam_theta_deg=CAMERA["theta"],
        fov_y_deg=CAMERA["fov"],
        camera_mode="mesh:zup",
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aa_samples=samples,
        aa_seed=SEED,
        denoise=DenoiseSettings(enabled=False, method="none"),
    )
    config.cam_target = [0.0, 0.0, 4.0]
    return f3d.TerrainRenderParams(config)


def render(samples, height, environment):
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    result = offline.render_offline(
        renderer,
        f3d.MaterialSet.terrain_default(),
        f3d.IBL.from_hdr(str(environment), intensity=0.96),
        params(samples),
        np.flipud(height),
        settings=f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=8),
    )
    renderer.end_offline_accumulation()
    assert result.metadata["samples_used"] == samples
    return (
        np.asarray(result.hdr_frame.to_numpy_f32(), np.float32)[..., :3],
        np.asarray(result.aov_frame.albedo(), np.float32),
        np.asarray(result.aov_frame.normal(), np.float32),
        np.asarray(result.aov_frame.depth(), np.float32),
        result.metadata,
    )


def project(beauty, albedo, depth, texture, city):
    width, height = SIZE
    target = np.array([0.0, 0.0, 4.0], np.float32)
    phi, theta = map(math.radians, (CAMERA["phi"], CAMERA["theta"]))
    eye = target + CAMERA["radius"] * np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        np.float32,
    )
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    py, px = np.mgrid[0:height, 0:width]
    tangent = math.tan(math.radians(CAMERA["fov"]) / 2)
    ray = (
        forward
        + (2 * (px + 0.5) / width - 1)[..., None] * tangent * (width / height) * right
        + (1 - 2 * (py + 0.5) / height)[..., None] * tangent * up
    )
    ray /= np.linalg.norm(ray, axis=2, keepdims=True)
    linear_depth = 0.1 + depth * (6000.0 - 0.1)
    position = eye + ray * (
        linear_depth / np.maximum(np.sum(ray * forward, axis=2), 1e-6)
    )[..., None]
    rows, columns = city.shape
    i = (np.clip(position[..., 0] / SPAN + 0.5, 0, 1) * (columns - 1)).astype(int)
    j = (np.clip(0.5 - position[..., 1] / SPAN, 0, 1) * (rows - 1)).astype(int)
    keep = (depth > 0) & city[j, i]
    lighting = np.clip(beauty / np.maximum(albedo, 0.02), 0.15, 5.0)
    lighting = np.clip(lighting / (np.mean(lighting[keep], axis=0) + 1e-6), 0.58, 1.42)
    image = np.ones((*depth.shape, 3), np.float32)
    image[keep] = np.clip(texture[j, i] * lighting, 0, 1)[keep]
    return (image * 255).astype(np.uint8)


def save_image(array, name):
    path = OUTPUT / name
    Image.fromarray(array).save(path)
    return path


def roughness(array):
    image = array.astype(np.float32) / 255.0
    return float((np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()) / 2)


def main():
    OUTPUT.mkdir(exist_ok=True)
    inputs = np.load(ROOT / "toronto_inputs.npz")
    height, texture, city = inputs["height"], inputs["texture"].astype(np.float32) / 255.0, inputs["city"]
    environment = ROOT / "environment.hdr"
    write_environment(environment)

    raw64 = render(64, height, environment)
    raw264 = render(264, height, environment)
    denoised = atrous_denoise(
        raw264[0],
        albedo=raw264[1],
        normal=raw264[2],
        depth=raw264[3],
        iterations=4,
        sigma_color=0.1,
        sigma_albedo=0.2,
        sigma_normal=0.1,
        sigma_depth=0.1,
    )

    maps = [
        project(raw64[0], raw64[1], raw64[3], texture, city),
        project(raw264[0], raw264[1], raw264[3], texture, city),
        project(denoised, raw264[1], raw264[3], texture, city),
    ]
    names = ["64-raw.png", "264-raw.png", "264-atrous.png"]
    for image, name in zip(maps, names):
        save_image(image, name)

    crop = (500, 280, 1000, 630)
    labels = ["64 RAW", "264 RAW", "264 + A-TROUS"]
    sheet = Image.new("RGB", (1500, 400), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    for index, (image, label) in enumerate(zip(maps, labels)):
        sheet.paste(Image.fromarray(image).crop(crop), (index * 500, 50))
        draw.text((index * 500 + 16, 12), label, font=font, fill=(25, 28, 32))
    sheet.save(OUTPUT / "comparison.png")

    crops = [image[crop[1] : crop[3], crop[0] : crop[2]].astype(np.float32) for image in maps]
    metrics = {
        "seed": SEED,
        "64_raw": raw64[4],
        "264_raw": raw264[4],
        "crop_mae_64_vs_264": float(np.mean(np.abs(crops[0] - crops[1]))),
        "crop_mae_264_vs_atrous": float(np.mean(np.abs(crops[1] - crops[2]))),
        "roughness": dict(zip(labels, map(roughness, crops))),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
