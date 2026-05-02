"""Required-thrust sizing for the Mach 6 cruise case."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CRUISE_MACH = 6.0
GAMMA_AIR = 1.4
R_AIR_J_PER_KG_K = 287.05
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_CSV = REPO_ROOT / "runs" / "route_visualization" / "data" / "spine_curve.csv"


@dataclass(frozen=True)
class ThrustRequiredEstimate:
    required_thrust_N: float
    drag_coefficient: float
    planform_area_m2: float
    mean_density_kg_m3: float
    mean_temperature_k: float
    cruise_mach: float
    average_velocity_m_s: float
    dynamic_pressure_Pa: float
    density_source: str
    drag_coefficient_source: str


def speed_of_sound_m_s(temperature_k: float) -> float:
    """Return the local speed of sound from static temperature."""
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive.")

    return math.sqrt(GAMMA_AIR * R_AIR_J_PER_KG_K * temperature_k)


def mach_velocity_m_s(mach: float, temperature_k: float) -> float:
    """Return flight speed for a Mach number at the supplied temperature."""
    if mach <= 0.0:
        raise ValueError("mach must be positive.")

    return mach * speed_of_sound_m_s(temperature_k)


def _mean_route_atmosphere_from_csv(csv_path: Path = DEFAULT_ROUTE_CSV) -> tuple[float, float, str]:
    density_sum = 0.0
    temperature_sum = 0.0
    sample_count = 0

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"density_kg_m3", "temperature_k"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} does not contain density_kg_m3 and temperature_k columns.")

        for row in reader:
            density_sum += float(row["density_kg_m3"])
            temperature_sum += float(row["temperature_k"])
            sample_count += 1

    if sample_count == 0:
        raise ValueError(f"{csv_path} does not contain route atmosphere samples.")

    return density_sum / sample_count, temperature_sum / sample_count, str(csv_path)


def _mean_route_atmosphere_from_script() -> tuple[float, float, str]:
    """Evaluate route_visualization.py directly when its CSV has not been generated."""
    from route_visualization import (
        FLYOVER_REGIONS,
        build_spine_curve,
        evaluate_density_profile_nrlmsis,
        optimize_flyover_route,
    )

    optimized_anchor_points, _ = optimize_flyover_route(FLYOVER_REGIONS)
    spine_lat, spine_lon = build_spine_curve(optimized_anchor_points)
    density_kg_m3, temperature_k = evaluate_density_profile_nrlmsis(spine_lat, spine_lon)
    return (
        float(density_kg_m3.mean()),
        float(temperature_k.mean()),
        "route_visualization.py",
    )


def mean_route_atmosphere() -> tuple[float, float, str]:
    """Return mean density and temperature from the route visualization output."""
    if DEFAULT_ROUTE_CSV.exists():
        return _mean_route_atmosphere_from_csv(DEFAULT_ROUTE_CSV)

    return _mean_route_atmosphere_from_script()


def _main_py_drag_inputs() -> tuple[float, float, str]:
    """Pull CD and reference area from the project baseline built in main.py."""
    from main import build_waverider

    waverider = build_waverider()
    if waverider.CD_total is None or waverider.inviscid_forces is None:
        raise RuntimeError("main.py did not produce CD_total and planform area.")

    return (
        float(waverider.CD_total),
        float(waverider.inviscid_forces["planform_area"]),
        "main.py",
    )


def Thruster_I_Hardly_Even_Know_Her(
    *,
    drag_coefficient: float | None = None,
    planform_area_m2: float | None = None,
    mean_density_kg_m3: float | None = None,
    mean_temperature_k: float | None = None,
    cruise_mach: float = DEFAULT_CRUISE_MACH,
) -> ThrustRequiredEstimate:
    """Estimate required installed thrust for steady, level Mach 6 cruise.

    Required thrust is set equal to drag:

        T_required = D = 0.5 * rho * V^2 * S_ref * CD

    By default, ``CD`` and ``S_ref`` come from ``main.py`` and route-average
    density/temperature come from ``route_visualization.py`` output.
    """
    if drag_coefficient is None or planform_area_m2 is None:
        main_cd, main_area, drag_source = _main_py_drag_inputs()
        if drag_coefficient is None:
            drag_coefficient = main_cd
        if planform_area_m2 is None:
            planform_area_m2 = main_area
    else:
        drag_source = "function arguments"

    if drag_coefficient < 0.0:
        raise ValueError("drag_coefficient must be non-negative.")
    if planform_area_m2 <= 0.0:
        raise ValueError("planform_area_m2 must be positive.")

    if mean_density_kg_m3 is None or mean_temperature_k is None:
        route_density, route_temperature, density_source = mean_route_atmosphere()
        if mean_density_kg_m3 is None:
            mean_density_kg_m3 = route_density
        if mean_temperature_k is None:
            mean_temperature_k = route_temperature
    else:
        density_source = "function arguments"

    if mean_density_kg_m3 < 0.0:
        raise ValueError("mean_density_kg_m3 must be non-negative.")

    average_velocity_m_s = mach_velocity_m_s(cruise_mach, mean_temperature_k)
    dynamic_pressure_Pa = 0.5 * mean_density_kg_m3 * average_velocity_m_s**2
    required_thrust_N = dynamic_pressure_Pa * planform_area_m2 * drag_coefficient

    return ThrustRequiredEstimate(
        required_thrust_N=required_thrust_N,
        drag_coefficient=drag_coefficient,
        planform_area_m2=planform_area_m2,
        mean_density_kg_m3=mean_density_kg_m3,
        mean_temperature_k=mean_temperature_k,
        cruise_mach=cruise_mach,
        average_velocity_m_s=average_velocity_m_s,
        dynamic_pressure_Pa=dynamic_pressure_Pa,
        density_source=density_source,
        drag_coefficient_source=drag_source,
    )


calculate_required_thrust = Thruster_I_Hardly_Even_Know_Her


if __name__ == "__main__":
    estimate = Thruster_I_Hardly_Even_Know_Her()
    print("Required Thrust Estimate")
    print(f"  Required thrust      = {estimate.required_thrust_N:,.1f} N")
    print(f"  CD source            = {estimate.drag_coefficient_source}")
    print(f"  CD                   = {estimate.drag_coefficient:.5f}")
    print(f"  Planform area        = {estimate.planform_area_m2:,.3f} m^2")
    print(f"  Density source       = {estimate.density_source}")
    print(f"  Mean density         = {estimate.mean_density_kg_m3:.6f} kg/m^3")
    print(f"  Mean temperature     = {estimate.mean_temperature_k:.2f} K")
    print(f"  Cruise Mach          = {estimate.cruise_mach:.1f}")
    print(f"  Average velocity     = {estimate.average_velocity_m_s:,.1f} m/s")
    print(f"  Dynamic pressure     = {estimate.dynamic_pressure_Pa:,.1f} Pa")
