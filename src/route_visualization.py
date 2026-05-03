
# This script gives an optimized route for our aircraft.
# It also allows us to determine the air density along the route.


r"""
Plan a minimum-distance overflight route from Uppsala to Singapore.

The route model treats Uppsala and Singapore as fixed points, while the
intermediate places of interest are modeled as flyover regions. The script:

1. Optimizes the crossing point inside each region to minimize total route length.
2. Builds a smooth cubic spine curve through the optimized anchors.
3. Evaluates NRLMSIS density along the route at 70,000 ft.
4. Saves map, globe, density visualizations, and CSV outputs for later analysis.

Run from the repository root:
    .\.venv\Scripts\python.exe src\route_visualization.py
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


EARTH_RADIUS_KM = 6371.0
FT_TO_M = 0.3048
CRUISE_ALTITUDE_FT = 70000.0
CRUISE_ALTITUDE_KM = CRUISE_ALTITUDE_FT * FT_TO_M / 1000.0
FLIGHT_PATH_RADIUS_KM = EARTH_RADIUS_KM + CRUISE_ALTITUDE_KM
REPO_ROOT = Path(__file__).resolve().parents[1]
EARTH_TEXTURE_CANDIDATES = [
    REPO_ROOT / "assets" / "earth" / "bluemarble-2048.png",
    REPO_ROOT / "assets" / "earth" / "bluemarble-1024.png",
]
GLOBE_N_LON = 361
GLOBE_N_LAT = 181
ORTHOGRAPHIC_IMAGE_SIZE = 1400
INTERACTIVE_MAX_POINTS = 220
SPINE_SAMPLE_COUNT = 1200
NRLMSIS_VERSION = 2.0
NRLMSIS_REFERENCE_DATES = np.array(
    [
        "2024-03-20T12:00",
        "2024-06-21T12:00",
        "2024-09-22T12:00",
        "2024-12-21T12:00",
    ],
    dtype="datetime64[m]",
)
NRLMSIS_F107 = 150.0
NRLMSIS_F107A = 150.0
NRLMSIS_AP_VECTOR = np.full(7, 4.0)


@dataclass(frozen=True)
class Waypoint:
    name: str
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True)
class FlyoverRegion:
    name: str
    lat_deg: float
    lon_deg: float
    radius_km: float


DEPARTURE = Waypoint("Uppsala", 59.8586, 17.6389)
DESTINATION = Waypoint("Singapore", 1.3521, 103.8198)

# Assumed circular flyover regions. These radii are editable and intentionally
# generous because the assignment wording says these only need to be flown over,
# not hit at an exact point.
FLYOVER_REGIONS = [
    FlyoverRegion("Gulf of Bothnia", 63.5, 20.0, 240.0),
    FlyoverRegion("Greenland", 72.0, -40.0, 1150.0),
    FlyoverRegion("Arctic Ocean", 83.0, -155.0, 1700.0),
    FlyoverRegion("Pacific Ocean", 31.0, 170.0, 2100.0),
    FlyoverRegion("South China Sea", 13.0, 114.0, 850.0),
]


def wrap_lon_deg(lon_deg: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(lon_deg) + 180.0) % 360.0 - 180.0


def sph_to_cart(lat_deg: np.ndarray | float, lon_deg: np.ndarray | float, radius: float = 1.0) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.column_stack((np.ravel(x), np.ravel(y), np.ravel(z)))


def cart_to_latlon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=float)
    xyz = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    lat = np.rad2deg(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lat, wrap_lon_deg(lon)


def central_angle_rad(a: Waypoint, b: Waypoint) -> float:
    lat1, lon1 = np.deg2rad([a.lat_deg, a.lon_deg])
    lat2, lon2 = np.deg2rad([b.lat_deg, b.lon_deg])
    return float(
        np.arccos(
            np.clip(
                np.sin(lat1) * np.sin(lat2)
                + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
                -1.0,
                1.0,
            )
        )
    )


def geodesic_distance_km(a: Waypoint, b: Waypoint, radius_km: float = EARTH_RADIUS_KM) -> float:
    return radius_km * central_angle_rad(a, b)


def cumulative_distance_km(points: list[Waypoint], radius_km: float = EARTH_RADIUS_KM) -> float:
    return sum(geodesic_distance_km(points[i], points[i + 1], radius_km=radius_km) for i in range(len(points) - 1))


def sampled_route_length_km(lat_deg: np.ndarray, lon_deg: np.ndarray, radius_km: float = EARTH_RADIUS_KM) -> float:
    total = 0.0
    for i in range(len(lat_deg) - 1):
        total += geodesic_distance_km(
            Waypoint("a", float(lat_deg[i]), float(lon_deg[i])),
            Waypoint("b", float(lat_deg[i + 1]), float(lon_deg[i + 1])),
            radius_km=radius_km,
        )
    return total


def flight_path_distance_km(surface_distance_km: float) -> float:
    return surface_distance_km * FLIGHT_PATH_RADIUS_KM / EARTH_RADIUS_KM


def cumulative_sampled_distances_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cumulative_surface_km = np.zeros(len(lat_deg), dtype=float)
    cumulative_flight_km = np.zeros(len(lat_deg), dtype=float)

    for i in range(1, len(lat_deg)):
        segment_surface_km = geodesic_distance_km(
            Waypoint("a", float(lat_deg[i - 1]), float(lon_deg[i - 1])),
            Waypoint("b", float(lat_deg[i]), float(lon_deg[i])),
        )
        cumulative_surface_km[i] = cumulative_surface_km[i - 1] + segment_surface_km
        cumulative_flight_km[i] = cumulative_flight_km[i - 1] + flight_path_distance_km(segment_surface_km)

    return cumulative_surface_km, cumulative_flight_km


def destination_point(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: np.ndarray | float,
    distance_km: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    lat1 = np.deg2rad(lat_deg)
    lon1 = np.deg2rad(lon_deg)
    bearing = np.deg2rad(bearing_deg)
    delta = np.asarray(distance_km, dtype=float) / EARTH_RADIUS_KM

    sin_lat2 = np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(bearing)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(lat1),
        np.cos(delta) - np.sin(lat1) * np.sin(lat2),
    )
    return np.rad2deg(lat2), wrap_lon_deg(np.rad2deg(lon2))


def initial_bearing_deg(a: Waypoint, b: Waypoint) -> float:
    lat1 = np.deg2rad(a.lat_deg)
    lat2 = np.deg2rad(b.lat_deg)
    dlon = np.deg2rad(b.lon_deg - a.lon_deg)

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float(wrap_lon_deg(np.rad2deg(np.arctan2(x, y))))


def circular_mean_deg(angles_deg: list[float]) -> float:
    angles_rad = np.deg2rad(angles_deg)
    return float(wrap_lon_deg(np.rad2deg(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))))


def great_circle_segment(start: Waypoint, end: Waypoint, n_points: int = 160) -> tuple[np.ndarray, np.ndarray]:
    a = sph_to_cart(start.lat_deg, start.lon_deg)[0]
    b = sph_to_cart(end.lat_deg, end.lon_deg)[0]

    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.isclose(omega, 0.0):
        lat = np.full(n_points, start.lat_deg)
        lon = np.full(n_points, start.lon_deg)
        return lat, lon

    t = np.linspace(0.0, 1.0, n_points)
    sin_omega = np.sin(omega)
    xyz = (
        np.sin((1.0 - t) * omega)[:, None] / sin_omega * a[None, :]
        + np.sin(t * omega)[:, None] / sin_omega * b[None, :]
    )
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    return cart_to_latlon(xyz)


def build_piecewise_route(points: list[Waypoint], n_points_per_leg: int = 160) -> tuple[np.ndarray, np.ndarray]:
    route_lat = []
    route_lon = []
    for i in range(len(points) - 1):
        lat, lon = great_circle_segment(points[i], points[i + 1], n_points=n_points_per_leg)
        if i > 0:
            lat = lat[1:]
            lon = lon[1:]
        route_lat.append(lat)
        route_lon.append(lon)
    return np.concatenate(route_lat), np.concatenate(route_lon)


def split_dateline(lat: np.ndarray, lon: np.ndarray, jump_deg: float = 180.0) -> list[tuple[np.ndarray, np.ndarray]]:
    split_idx = np.where(np.abs(np.diff(lon)) > jump_deg)[0] + 1
    lat_parts = np.split(lat, split_idx)
    lon_parts = np.split(lon, split_idx)
    return list(zip(lat_parts, lon_parts))


def split_visible_segments(x: np.ndarray, y: np.ndarray, visible: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    break_idx = np.where(~visible[:-1] | ~visible[1:])[0] + 1
    x_parts = np.split(x, break_idx)
    y_parts = np.split(y, break_idx)
    vis_parts = np.split(visible, break_idx)
    segments = []
    for x_part, y_part, vis_part in zip(x_parts, y_parts, vis_parts):
        if len(x_part) >= 2 and np.all(vis_part):
            segments.append((x_part, y_part))
    return segments


def load_earth_texture(texture_paths: list[Path] = EARTH_TEXTURE_CANDIDATES) -> np.ndarray | None:
    texture_path = next((path for path in texture_paths if path.exists()), None)
    if texture_path is None:
        return None

    texture = plt.imread(texture_path)
    if texture.dtype.kind in {"u", "i"}:
        texture = texture.astype(np.float32) / 255.0
    else:
        texture = texture.astype(np.float32)

    if texture.shape[-1] == 4:
        texture = texture[..., :3]

    return texture


def sample_texture(texture: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    n_rows, n_cols = texture.shape[:2]
    lon_wrapped = np.mod(lon_deg + 180.0, 360.0)
    col = lon_wrapped / 360.0 * n_cols
    row = ((90.0 - lat_deg) / 180.0) * (n_rows - 1)

    col0 = np.floor(col).astype(int) % n_cols
    col1 = (col0 + 1) % n_cols
    row0 = np.clip(np.floor(row).astype(int), 0, n_rows - 1)
    row1 = np.clip(row0 + 1, 0, n_rows - 1)

    dx = (col - np.floor(col))[..., None]
    dy = (row - np.floor(row))[..., None]

    c00 = texture[row0, col0]
    c10 = texture[row0, col1]
    c01 = texture[row1, col0]
    c11 = texture[row1, col1]

    c0 = (1.0 - dx) * c00 + dx * c10
    c1 = (1.0 - dx) * c01 + dx * c11
    return (1.0 - dy) * c0 + dy * c1


def shade_colors(colors: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    light_dir = np.array([1.0, -0.6, 0.8], dtype=float)
    light_dir /= np.linalg.norm(light_dir)

    normals = np.stack((x, y, z), axis=-1)
    intensity = np.clip(np.sum(normals * light_dir[None, None, :], axis=-1), -0.2, 1.0)
    intensity = 0.35 + 0.65 * (intensity + 0.2) / 1.2

    shaded = colors.copy()
    shaded[..., :3] = np.clip(shaded[..., :3] * intensity[..., None], 0.0, 1.0)
    return shaded


def compute_view_center(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[float, float]:
    xyz = sph_to_cart(lat_deg, lon_deg)
    mean_vec = xyz.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm < 1e-12:
        return 20.0, 90.0

    mean_vec /= mean_norm
    center_lat = float(np.rad2deg(np.arcsin(mean_vec[2])))
    center_lon = float(np.rad2deg(np.arctan2(mean_vec[1], mean_vec[0])))
    return center_lat, center_lon


def orthographic_project(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat_0_deg: float,
    lon_0_deg: float,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat_0 = np.deg2rad(lat_0_deg)
    lon_0 = np.deg2rad(lon_0_deg)

    lon_rel = lon - lon_0
    cos_c = np.sin(lat_0) * np.sin(lat) + np.cos(lat_0) * np.cos(lat) * np.cos(lon_rel)
    visible = cos_c >= 0.0

    x = radius * np.cos(lat) * np.sin(lon_rel)
    y = radius * (
        np.cos(lat_0) * np.sin(lat)
        - np.sin(lat_0) * np.cos(lat) * np.cos(lon_rel)
    )
    return x, y, visible


def render_orthographic_texture(
    texture: np.ndarray | None,
    lat_0_deg: float,
    lon_0_deg: float,
    image_size: int = ORTHOGRAPHIC_IMAGE_SIZE,
) -> np.ndarray:
    bg_rgba = np.zeros((image_size, image_size, 4), dtype=np.float32)
    bg_rgba[..., :3] = np.array([5, 7, 13], dtype=np.float32) / 255.0

    axis = np.linspace(-1.0, 1.0, image_size)
    xx, yy = np.meshgrid(axis, axis)
    rho = np.sqrt(xx**2 + yy**2)
    inside = rho <= 1.0

    if texture is None:
        globe_rgb = np.zeros((image_size, image_size, 3), dtype=np.float32)
        globe_rgb[...] = np.array([168, 213, 226], dtype=np.float32) / 255.0
        globe_rgba = np.concatenate([globe_rgb, inside[..., None].astype(np.float32)], axis=-1)
        bg_rgba[inside] = globe_rgba[inside]
        return bg_rgba

    lat_0 = np.deg2rad(lat_0_deg)
    lon_0 = np.deg2rad(lon_0_deg)

    rho_safe = np.where(inside & (rho > 1e-12), rho, 1.0)
    c = np.arcsin(np.clip(rho, 0.0, 1.0))
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    lat = np.empty_like(xx)
    lon = np.empty_like(xx)
    lat[:] = lat_0
    lon[:] = lon_0

    lat[inside] = np.arcsin(
        cos_c[inside] * np.sin(lat_0)
        + (yy[inside] * sin_c[inside] * np.cos(lat_0) / rho_safe[inside])
    )
    lon[inside] = lon_0 + np.arctan2(
        xx[inside] * sin_c[inside],
        rho_safe[inside] * np.cos(lat_0) * cos_c[inside]
        - yy[inside] * np.sin(lat_0) * sin_c[inside],
    )

    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    globe_rgb = sample_texture(texture, lat_deg, lon_deg)
    zz = np.sqrt(np.clip(1.0 - xx**2 - yy**2, 0.0, 1.0))
    globe_rgb = shade_colors(globe_rgb, xx, yy, zz)

    alpha = inside.astype(np.float32)
    rgba = np.concatenate([globe_rgb, alpha[..., None]], axis=-1)
    bg_rgba[inside] = rgba[inside]
    return bg_rgba


def region_boundary_latlon(region: FlyoverRegion, n_points: int = 241) -> tuple[np.ndarray, np.ndarray]:
    bearings = np.linspace(-180.0, 180.0, n_points)
    lat, lon = destination_point(region.lat_deg, region.lon_deg, bearings, region.radius_km)
    return lat, lon


def region_center_waypoint(region: FlyoverRegion) -> Waypoint:
    return Waypoint(region.name, region.lat_deg, region.lon_deg)


def params_to_waypoints(params: np.ndarray, regions: list[FlyoverRegion]) -> list[Waypoint]:
    waypoints = []
    for region, radius_frac, bearing_deg in zip(regions, params[0::2], params[1::2]):
        lat, lon = destination_point(region.lat_deg, region.lon_deg, bearing_deg, radius_frac * region.radius_km)
        waypoints.append(Waypoint(region.name, float(lat), float(lon)))
    return waypoints


def build_route_from_params(params: np.ndarray, regions: list[FlyoverRegion]) -> list[Waypoint]:
    return [DEPARTURE] + params_to_waypoints(params, regions) + [DESTINATION]


def route_objective(params: np.ndarray, regions: list[FlyoverRegion]) -> float:
    return cumulative_distance_km(build_route_from_params(params, regions))


def seeded_initial_guesses(regions: list[FlyoverRegion]) -> list[np.ndarray]:
    center_path = [DEPARTURE] + [region_center_waypoint(region) for region in regions] + [DESTINATION]
    base_bearings = []

    for idx, region in enumerate(regions, start=1):
        region_center = center_path[idx]
        prev_point = center_path[idx - 1]
        next_point = center_path[idx + 1]
        base_bearings.append(
            circular_mean_deg(
                [
                    initial_bearing_deg(region_center, prev_point),
                    initial_bearing_deg(region_center, next_point),
                ]
            )
        )

    guesses = []
    for alpha in (0.0, 0.35, 0.70, 1.0):
        x0 = np.zeros(2 * len(regions), dtype=float)
        x0[0::2] = alpha
        x0[1::2] = base_bearings
        guesses.append(x0)

    rng = np.random.default_rng(1226)
    for _ in range(10):
        x0 = np.zeros(2 * len(regions), dtype=float)
        x0[0::2] = rng.uniform(0.0, 1.0, size=len(regions))
        x0[1::2] = wrap_lon_deg(np.array(base_bearings) + rng.normal(0.0, 35.0, size=len(regions)))
        guesses.append(x0)

    return guesses


def optimize_flyover_route(regions: list[FlyoverRegion]) -> tuple[list[Waypoint], float]:
    bounds = []
    for _ in regions:
        bounds.extend([(0.0, 1.0), (-180.0, 180.0)])

    best_result = None
    for x0 in seeded_initial_guesses(regions):
        result = minimize(
            route_objective,
            x0,
            args=(regions,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("Route optimization failed to start.")

    best_route = build_route_from_params(best_result.x, regions)
    return best_route, float(best_result.fun)


def build_spine_curve(points: list[Waypoint], n_samples: int = SPINE_SAMPLE_COUNT) -> tuple[np.ndarray, np.ndarray]:
    anchor_lat = np.array([point.lat_deg for point in points], dtype=float)
    anchor_lon = np.array([point.lon_deg for point in points], dtype=float)
    anchor_xyz = sph_to_cart(anchor_lat, anchor_lon)

    s = np.zeros(len(points), dtype=float)
    for i in range(1, len(points)):
        s[i] = s[i - 1] + geodesic_distance_km(points[i - 1], points[i])

    spline_x = CubicSpline(s, anchor_xyz[:, 0], bc_type="natural")
    spline_y = CubicSpline(s, anchor_xyz[:, 1], bc_type="natural")
    spline_z = CubicSpline(s, anchor_xyz[:, 2], bc_type="natural")

    s_eval = np.linspace(s[0], s[-1], n_samples)
    xyz = np.column_stack((spline_x(s_eval), spline_y(s_eval), spline_z(s_eval)))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    return cart_to_latlon(xyz)


def downsample_route(lat_deg: np.ndarray, lon_deg: np.ndarray, max_points: int = INTERACTIVE_MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    if len(lat_deg) <= max_points:
        return lat_deg, lon_deg

    idx = np.linspace(0, len(lat_deg) - 1, max_points).astype(int)
    idx = np.unique(idx)
    return lat_deg[idx], lon_deg[idx]


def ensure_pymsis_param_file(version: float = NRLMSIS_VERSION) -> None:
    from pymsis import __file__ as pymsis_file

    package_dir = Path(pymsis_file).resolve().parent
    site_packages_dir = package_dir.parent

    if float(version) == 2.0:
        filename = "msis2.0.parm"
    elif float(version) == 2.1:
        filename = "msis21.parm"
    else:
        return

    source_path = package_dir / filename
    legacy_path = site_packages_dir / f"pyms{filename}"

    if source_path.exists() and not legacy_path.exists():
        shutil.copyfile(source_path, legacy_path)


def evaluate_density_profile_nrlmsis(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_km: float = CRUISE_ALTITUDE_KM,
) -> tuple[np.ndarray, np.ndarray]:
    from pymsis import Variable, msis

    ensure_pymsis_param_file(version=NRLMSIS_VERSION)

    n_points = len(lat_deg)
    altitudes_km = np.full(n_points, altitude_km, dtype=float)

    seasonal_densities = []
    seasonal_temperatures = []
    for reference_date in NRLMSIS_REFERENCE_DATES:
        dates = np.full(n_points, reference_date, dtype="datetime64[m]")
        f107s = np.full(n_points, NRLMSIS_F107, dtype=float)
        f107as = np.full(n_points, NRLMSIS_F107A, dtype=float)
        aps = np.tile(NRLMSIS_AP_VECTOR, (n_points, 1)).astype(float)

        output = msis.run(
            dates,
            lon_deg,
            lat_deg,
            altitudes_km,
            f107s=f107s,
            f107as=f107as,
            aps=aps,
            version=NRLMSIS_VERSION,
        )
        seasonal_densities.append(output[:, Variable.MASS_DENSITY])
        seasonal_temperatures.append(output[:, Variable.TEMPERATURE])

    density_kg_m3 = np.mean(np.vstack(seasonal_densities), axis=0)
    temperature_k = np.mean(np.vstack(seasonal_temperatures), axis=0)
    return density_kg_m3, temperature_k


def plot_density_vs_distance(
    distance_km: np.ndarray,
    density_kg_m3: np.ndarray,
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("#f6f4ed")
    ax.set_facecolor("white")

    ax.plot(distance_km, density_kg_m3, color="#0f4c81", lw=2.5)
    ax.fill_between(distance_km, density_kg_m3, color="#7fb7df", alpha=0.25)
    ax.set_xlabel("Distance Traveled at 70,000 ft [km]")
    ax.set_ylabel(r"Density [kg/m$^3$]")
    ax.set_title("NRLMSIS 2.0 Mean Density Along the Optimized Route")
    ax.grid(True, color="#d7dde3", linewidth=0.8)
    y_min = float(density_kg_m3.min())
    y_max = float(density_kg_m3.max())
    padding = 0.15 * (y_max - y_min if y_max > y_min else y_max)
    ax.set_ylim(y_min - padding, y_max + padding)

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    return fig, ax


def plot_route_map(
    anchor_points: list[Waypoint],
    spine_lat: np.ndarray,
    spine_lon: np.ndarray,
    regions: list[FlyoverRegion],
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    fig.patch.set_facecolor("#f6f4ed")
    ax.set_facecolor("#dfeaf4")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Optimized Sweden-to-Singapore Overflight Route")

    for lon in range(-180, 181, 30):
        ax.plot([lon, lon], [-90, 90], color="white", lw=0.8, alpha=0.8, zorder=0)
    for lat in range(-90, 91, 15):
        ax.plot([-180, 180], [lat, lat], color="white", lw=0.8, alpha=0.8, zorder=0)

    for region in regions:
        boundary_lat, boundary_lon = region_boundary_latlon(region)
        for seg_lat, seg_lon in split_dateline(boundary_lat, boundary_lon):
            ax.plot(seg_lon, seg_lat, color="#4aa3d8", lw=1.4, ls="--", alpha=0.9, zorder=1)

    anchor_leg_lat, anchor_leg_lon = build_piecewise_route(anchor_points, n_points_per_leg=120)
    for seg_lat, seg_lon in split_dateline(anchor_leg_lat, anchor_leg_lon):
        ax.plot(seg_lon, seg_lat, color="white", lw=4.5, alpha=0.45, zorder=2)
        ax.plot(seg_lon, seg_lat, color="#7d8597", lw=1.5, ls="--", alpha=0.95, zorder=3)

    for seg_lat, seg_lon in split_dateline(spine_lat, spine_lon):
        ax.plot(seg_lon, seg_lat, color="white", lw=5.0, alpha=0.45, zorder=4)
        ax.plot(seg_lon, seg_lat, color="#d1495b", lw=2.8, zorder=5)

    endpoint_lons = np.array([anchor_points[0].lon_deg, anchor_points[-1].lon_deg])
    endpoint_lats = np.array([anchor_points[0].lat_deg, anchor_points[-1].lat_deg])
    flyover_lons = np.array([point.lon_deg for point in anchor_points[1:-1]])
    flyover_lats = np.array([point.lat_deg for point in anchor_points[1:-1]])

    ax.scatter(endpoint_lons, endpoint_lats, s=75, color="#1f3c88", edgecolors="white", linewidths=1.0, zorder=6)
    ax.scatter(flyover_lons, flyover_lats, s=60, color="#ff7aa2", edgecolors="white", linewidths=0.9, zorder=6)

    for idx, point in enumerate(anchor_points, start=1):
        ax.text(
            point.lon_deg + 2.5,
            point.lat_deg + (2.0 if point.lat_deg < 75 else -4.0),
            f"{idx}. {point.name}",
            fontsize=9,
            color="#102542",
            zorder=7,
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    return fig, ax


def plot_route_orthographic_map(
    anchor_points: list[Waypoint],
    spine_lat: np.ndarray,
    spine_lon: np.ndarray,
    regions: list[FlyoverRegion],
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    center_lat, center_lon = compute_view_center(spine_lat, spine_lon)
    texture = load_earth_texture()
    globe_rgba = render_orthographic_texture(texture, center_lat, center_lon)

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    fig.patch.set_facecolor("#05070d")
    ax.set_facecolor("#05070d")
    ax.imshow(globe_rgba, extent=(-1, 1, -1, 1), origin="lower", interpolation="bilinear")
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, lw=1.4, color="#d9ecff", alpha=0.95))

    for region in regions:
        boundary_lat, boundary_lon = region_boundary_latlon(region)
        bx, by, bvis = orthographic_project(boundary_lat, boundary_lon, center_lat, center_lon, radius=1.0)
        for x_seg, y_seg in split_visible_segments(bx, by, bvis):
            ax.plot(x_seg, y_seg, color="#66c5ff", lw=1.2, ls="--", alpha=0.85, zorder=2)

    anchor_leg_lat, anchor_leg_lon = build_piecewise_route(anchor_points, n_points_per_leg=120)
    leg_x, leg_y, leg_vis = orthographic_project(anchor_leg_lat, anchor_leg_lon, center_lat, center_lon, radius=1.0)
    for x_seg, y_seg in split_visible_segments(leg_x, leg_y, leg_vis):
        ax.plot(x_seg, y_seg, color="white", lw=4.2, alpha=0.35, zorder=3)
        ax.plot(x_seg, y_seg, color="#7d8597", lw=1.3, ls="--", alpha=0.9, zorder=4)

    route_x, route_y, route_vis = orthographic_project(spine_lat, spine_lon, center_lat, center_lon, radius=1.0)
    for x_seg, y_seg in split_visible_segments(route_x, route_y, route_vis):
        ax.plot(x_seg, y_seg, color="white", lw=5.0, alpha=0.35, zorder=5)
        ax.plot(x_seg, y_seg, color="#ff5d73", lw=3.0, zorder=6)

    point_lats = np.array([point.lat_deg for point in anchor_points], dtype=float)
    point_lons = np.array([point.lon_deg for point in anchor_points], dtype=float)
    point_x, point_y, point_vis = orthographic_project(point_lats, point_lons, center_lat, center_lon, radius=1.0)

    ax.scatter(
        point_x[point_vis],
        point_y[point_vis],
        s=58,
        color="#ff7aa2",
        edgecolors="white",
        linewidths=0.9,
        zorder=7,
    )

    for idx, point in enumerate(anchor_points, start=1):
        if not point_vis[idx - 1]:
            continue
        ax.text(
            point_x[idx - 1] + 0.03,
            point_y[idx - 1] + 0.03,
            f"{idx}. {point.name}",
            color="white",
            fontsize=10,
            zorder=8,
        )

    ax.set_title("Optimized Route on an Orthographic Globe", color="white", pad=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    if save_path is not None:
        fig.savefig(save_path, dpi=280, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig, ax


def plot_route_globe(
    anchor_points: list[Waypoint],
    spine_lat: np.ndarray,
    spine_lon: np.ndarray,
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    fig.patch.set_facecolor("#05070d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Optimized Route on a Globe", color="white", pad=18)
    ax.set_box_aspect((1, 1, 1))

    u = np.linspace(0.0, 2.0 * np.pi, GLOBE_N_LON)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, GLOBE_N_LAT)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(vv) * np.cos(uu)
    y = np.cos(vv) * np.sin(uu)
    z = np.sin(vv)

    texture = load_earth_texture()
    if texture is not None:
        lon_deg = np.rad2deg(np.arctan2(y, x))
        lat_deg = np.rad2deg(np.arcsin(z))
        earth_colors = sample_texture(texture, lat_deg, lon_deg)
        earth_colors = shade_colors(earth_colors, x, y, z)
        ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            facecolors=earth_colors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            zorder=0,
        )
        ax.plot_surface(
            1.018 * x,
            1.018 * y,
            1.018 * z,
            rstride=3,
            cstride=3,
            color="#80d8ff",
            linewidth=0.0,
            alpha=0.08,
            shade=False,
            zorder=1,
        )
    else:
        ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            color="#a8d5e2",
            edgecolor="#b7c8cf",
            linewidth=0.15,
            alpha=0.5,
            zorder=0,
        )

    route_xyz = sph_to_cart(spine_lat, spine_lon, radius=1.02)
    ax.plot(route_xyz[:, 0], route_xyz[:, 1], route_xyz[:, 2], color="#ff5d73", lw=3.0, zorder=4)

    anchor_xyz = sph_to_cart(
        np.array([point.lat_deg for point in anchor_points]),
        np.array([point.lon_deg for point in anchor_points]),
        radius=1.03,
    )
    ax.scatter(
        anchor_xyz[:, 0],
        anchor_xyz[:, 1],
        anchor_xyz[:, 2],
        s=48,
        color="#ff7aa2",
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        zorder=5,
    )

    for idx, (point, xyz) in enumerate(zip(anchor_points, anchor_xyz), start=1):
        ax.text(
            xyz[0] * 1.08,
            xyz[1] * 1.08,
            xyz[2] * 1.08,
            f"{idx}. {point.name}",
            color="white",
            fontsize=9,
            zorder=6,
        )

    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, dpi=320, bbox_inches="tight")

    return fig, ax


def plot_route_plotly_globe(
    anchor_points: list[Waypoint],
    route_lat: np.ndarray,
    route_lon: np.ndarray,
    save_path: str | None = None,
) -> go.Figure:
    point_labels = [f"{idx}. {point.name}" for idx, point in enumerate(anchor_points, start=1)]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=route_lat,
            lon=route_lon,
            mode="lines",
            line=dict(width=4, color="#d1495b"),
            hoverinfo="skip",
            name="Spine curve",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[point.lat_deg for point in anchor_points],
            lon=[point.lon_deg for point in anchor_points],
            mode="markers",
            text=point_labels,
            marker=dict(size=8, color="#1f3c88", line=dict(width=1, color="white")),
            hovertemplate="%{text}<br>lat=%{lat:.2f} deg<br>lon=%{lon:.2f} deg<extra></extra>",
            name="Optimized anchors",
        )
    )

    fig.update_geos(
        projection_type="orthographic",
        resolution=110,
        showland=True,
        landcolor="#d7d3c8",
        showocean=True,
        oceancolor="#9ecae1",
        showlakes=False,
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="white",
        coastlinewidth=0.8,
        lataxis=dict(showgrid=False),
        lonaxis=dict(showgrid=False),
        bgcolor="#f6f4ed",
    )
    fig.update_layout(
        title="Optimized Route on an Interactive Orthographic Globe",
        paper_bgcolor="#f6f4ed",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    if save_path is not None:
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            config=dict(displaylogo=False, responsive=True),
        )

    return fig


def write_anchor_points_csv(save_path: str, points: list[Waypoint], regions: list[FlyoverRegion]) -> None:
    radius_lookup = {region.name: region.radius_km for region in regions}
    with open(save_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["index", "name", "role", "lat_deg", "lon_deg", "region_radius_km"])
        for idx, point in enumerate(points, start=1):
            if idx == 1:
                role = "departure"
            elif idx == len(points):
                role = "destination"
            else:
                role = "optimized_flyover"
            writer.writerow(
                [
                    idx,
                    point.name,
                    role,
                    f"{point.lat_deg:.6f}",
                    f"{point.lon_deg:.6f}",
                    f"{radius_lookup.get(point.name, 0.0):.1f}",
                ]
            )


def write_spine_curve_csv(
    save_path: str,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    density_kg_m3: np.ndarray | None = None,
    temperature_k: np.ndarray | None = None,
) -> None:
    cumulative_surface_km, cumulative_flight_km = cumulative_sampled_distances_km(lat_deg, lon_deg)

    with open(save_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        header = [
            "index",
            "lat_deg",
            "lon_deg",
            "cumulative_surface_distance_km",
            "cumulative_flight_distance_km",
        ]
        if density_kg_m3 is not None:
            header.append("density_kg_m3")
        if temperature_k is not None:
            header.append("temperature_k")
        writer.writerow(header)

        for idx in range(len(lat_deg)):
            row = [
                idx + 1,
                f"{lat_deg[idx]:.6f}",
                f"{lon_deg[idx]:.6f}",
                f"{cumulative_surface_km[idx]:.3f}",
                f"{cumulative_flight_km[idx]:.3f}",
            ]
            if density_kg_m3 is not None:
                row.append(f"{density_kg_m3[idx]:.8f}")
            if temperature_k is not None:
                row.append(f"{temperature_k[idx]:.3f}")
            writer.writerow(row)


def main() -> None:
    output_plot_dir = REPO_ROOT / "runs" / "route_visualization" / "plots"
    output_data_dir = REPO_ROOT / "runs" / "route_visualization" / "data"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    centerline_points = [DEPARTURE] + [region_center_waypoint(region) for region in FLYOVER_REGIONS] + [DESTINATION]
    centerline_surface_km = cumulative_distance_km(centerline_points)

    optimized_anchor_points, optimized_piecewise_surface_km = optimize_flyover_route(FLYOVER_REGIONS)
    spine_lat, spine_lon = build_spine_curve(optimized_anchor_points, n_samples=SPINE_SAMPLE_COUNT)
    interactive_lat, interactive_lon = downsample_route(spine_lat, spine_lon, max_points=INTERACTIVE_MAX_POINTS)
    spline_surface_km = sampled_route_length_km(spine_lat, spine_lon)
    centerline_flight_km = flight_path_distance_km(centerline_surface_km)
    optimized_piecewise_flight_km = flight_path_distance_km(optimized_piecewise_surface_km)
    spline_flight_km = flight_path_distance_km(spline_surface_km)
    _, cumulative_flight_km = cumulative_sampled_distances_km(spine_lat, spine_lon)
    density_kg_m3, temperature_k = evaluate_density_profile_nrlmsis(spine_lat, spine_lon)

    map_path = output_plot_dir / "route_map.png"
    globe_path = output_plot_dir / "route_globe.png"
    ortho_path = output_plot_dir / "route_orthographic.png"
    interactive_globe_path = output_plot_dir / "route_globe_interactive.html"
    density_plot_path = output_plot_dir / "density_vs_distance.png"
    anchors_csv_path = output_data_dir / "optimized_anchor_points.csv"
    spine_csv_path = output_data_dir / "spine_curve.csv"

    plot_route_map(optimized_anchor_points, spine_lat, spine_lon, FLYOVER_REGIONS, save_path=map_path)
    plot_route_globe(optimized_anchor_points, spine_lat, spine_lon, save_path=globe_path)
    plot_route_orthographic_map(optimized_anchor_points, spine_lat, spine_lon, FLYOVER_REGIONS, save_path=ortho_path)
    plot_route_plotly_globe(optimized_anchor_points, interactive_lat, interactive_lon, save_path=interactive_globe_path)
    plot_density_vs_distance(cumulative_flight_km, density_kg_m3, save_path=density_plot_path)
    write_anchor_points_csv(anchors_csv_path, optimized_anchor_points, FLYOVER_REGIONS)
    write_spine_curve_csv(spine_csv_path, spine_lat, spine_lon, density_kg_m3=density_kg_m3, temperature_k=temperature_k)

    textured_globe = "yes" if any(path.exists() for path in EARTH_TEXTURE_CANDIDATES) else "no"
    print("Route assumption:")
    print("  Intermediate flyover regions are traversed in the order listed in the assignment narrative.")
    print("  Each flyover region is modeled as a circular area centered on the coordinates below.")
    print(f"  Flight distance is evaluated at a constant altitude of {CRUISE_ALTITUDE_FT:,.0f} ft.")
    print("  NRLMSIS density is computed as a climatological mean of four seasonal reference dates.")
    print(f"  NRLMSIS settings: version = {NRLMSIS_VERSION:.1f}, F10.7 = {NRLMSIS_F107:.1f}, F10.7a = {NRLMSIS_F107A:.1f}, Ap = {NRLMSIS_AP_VECTOR[0]:.1f}")
    print("\nFlyover region model:")
    for region in FLYOVER_REGIONS:
        print(
            f"  {region.name:<16} center = ({region.lat_deg:7.3f}, {region.lon_deg:8.3f}) deg"
            f"   radius = {region.radius_km:6.1f} km"
        )

    print("\nOptimized anchor points:")
    for idx, point in enumerate(optimized_anchor_points, start=1):
        print(f"  {idx}. {point.name:<16} lat = {point.lat_deg:8.3f} deg, lon = {point.lon_deg:9.3f} deg")

    print(f"\nCenter-to-center route length on surface = {centerline_surface_km:,.0f} km")
    print(f"Center-to-center route length at 70,000 ft = {centerline_flight_km:,.0f} km")
    print(f"Optimized piecewise route length on surface = {optimized_piecewise_surface_km:,.0f} km")
    print(f"Optimized piecewise route length at 70,000 ft = {optimized_piecewise_flight_km:,.0f} km")
    print(f"Smooth spine length on surface (sampled) = {spline_surface_km:,.0f} km")
    print(f"Total distance traveled along smooth spine at 70,000 ft = {spline_flight_km:,.0f} km")
    print(f"Mean NRLMSIS density along route at 70,000 ft = {density_kg_m3.mean():.5f} kg/m^3")
    print(f"Density range along route at 70,000 ft = [{density_kg_m3.min():.5f}, {density_kg_m3.max():.5f}] kg/m^3")
    print(f"\nTOTAL AIRCRAFT DISTANCE COVERED = {spline_flight_km:,.0f} km")
    print(f"Textured Earth globe available = {textured_globe}")
    print(f"Saved map view            to {map_path}")
    print(f"Saved globe view          to {globe_path}")
    print(f"Saved orthographic globe  to {ortho_path}")
    print(f"Saved interactive globe   to {interactive_globe_path}")
    print(f"Saved density plot        to {density_plot_path}")
    print(f"Saved optimized anchors   to {anchors_csv_path}")
    print(f"Saved spine curve         to {spine_csv_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
