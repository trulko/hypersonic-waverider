"""Discrete optimizer for fixed-geometry Breguet input assumptions."""

from __future__ import annotations

from dataclasses import dataclass

from Breguet import calculate_breguet_range_estimate
from engine_sizing import estimate_engine_sizing


LB_TO_KG = 0.45359237
ALLOWED_FUEL_FRACTION = 0.15
THRUST_MARGIN = 1.20
MAX_THRUST_PER_ENGINE_N = 60000.0
ENGINE_COUNT_OPTIONS = (2, 4, 6, 8, 10, 12, 14, 16)
ISP_SAMPLE_COUNT = 31

# X-51A reference: JP-7 fueled SJY61 scramjet with approx. 270 lb JP-7 capacity.
X51A_FUEL_DENSITY_KG_M3 = 803.0
X51A_FUEL_CAPACITY_KG = 270.0 * LB_TO_KG
X51A_FUEL_VOLUME_M3 = X51A_FUEL_CAPACITY_KG / X51A_FUEL_DENSITY_KG_M3


@dataclass(frozen=True)
class FuelOption:
    name: str
    density_kg_m3: float
    min_isp_s: float
    max_isp_s: float


@dataclass(frozen=True)
class BreguetOptimizationCase:
    fuel: FuelOption
    engine_count: int
    specific_impulse_s: float
    fuel_volume_m3: float
    fuel_fraction: float
    x51a_fuel_volume_equivalent: float
    available_thrust_N: float
    required_thrust_with_margin_N: float
    engine_sizing: object
    estimate: object


FUEL_OPTIONS = (
    FuelOption("JP-7", 803.0, 1000.0, 2500.0),
    FuelOption("liquid methane", 422.0, 1000.0, 2800.0),
    FuelOption("liquid hydrogen", 70.8, 1500.0, 4000.0),
)


def estimate_fuel_storage_volume_m3(
    fuel_mass_kg: float,
    fuel_density_kg_m3: float = X51A_FUEL_DENSITY_KG_M3,
) -> float:
    """Return fuel storage volume from fuel mass and density."""
    if fuel_mass_kg < 0.0:
        raise ValueError("fuel_mass_kg must be non-negative.")
    if fuel_density_kg_m3 <= 0.0:
        raise ValueError("fuel_density_kg_m3 must be positive.")

    return fuel_mass_kg / fuel_density_kg_m3


def mass_fraction(component_mass_kg: float, total_mass_kg: float) -> float:
    """Return component mass as a fraction of total mass."""
    if total_mass_kg <= 0.0:
        raise ValueError("total_mass_kg must be positive.")

    return component_mass_kg / total_mass_kg


def isp_samples(fuel: FuelOption, sample_count: int = ISP_SAMPLE_COUNT) -> list[float]:
    """Return evenly spaced ISP values for a fuel/engine option."""
    if sample_count <= 1:
        return [fuel.max_isp_s]

    step = (fuel.max_isp_s - fuel.min_isp_s) / (sample_count - 1)
    return [fuel.min_isp_s + step * idx for idx in range(sample_count)]


def optimize_breguet_inputs(
    *,
    volume_m3: float,
    lift_to_drag: float,
    required_thrust_N: float,
) -> tuple[BreguetOptimizationCase | None, list[BreguetOptimizationCase]]:
    """Sweep fuel, engine count, and ISP and return the lightest feasible case."""
    feasible_cases = []
    required_thrust_with_margin_N = required_thrust_N * THRUST_MARGIN

    for fuel in FUEL_OPTIONS:
        for engine_count in ENGINE_COUNT_OPTIONS:
            available_thrust_N = engine_count * MAX_THRUST_PER_ENGINE_N
            if available_thrust_N < required_thrust_with_margin_N:
                continue

            engine_sizing = estimate_engine_sizing(
                required_thrust_N=required_thrust_N,
                engine_count=engine_count,
            )

            for specific_impulse_s in isp_samples(fuel):
                estimate = calculate_breguet_range_estimate(
                    volume_m3=volume_m3,
                    powerplant_mass_kg=engine_sizing.powerplant_mass_kg,
                    lift_to_drag=lift_to_drag,
                    specific_impulse_s=specific_impulse_s,
                    engine_count=engine_count,
                )
                fuel_volume_m3 = estimate_fuel_storage_volume_m3(
                    estimate.fuel_mass_kg,
                    fuel.density_kg_m3,
                )
                fuel_fraction = fuel_volume_m3 / volume_m3
                if fuel_fraction > ALLOWED_FUEL_FRACTION:
                    continue

                feasible_cases.append(
                    BreguetOptimizationCase(
                        fuel=fuel,
                        engine_count=engine_count,
                        specific_impulse_s=specific_impulse_s,
                        fuel_volume_m3=fuel_volume_m3,
                        fuel_fraction=fuel_fraction,
                        x51a_fuel_volume_equivalent=fuel_volume_m3 / X51A_FUEL_VOLUME_M3,
                        available_thrust_N=available_thrust_N,
                        required_thrust_with_margin_N=required_thrust_with_margin_N,
                        engine_sizing=engine_sizing,
                        estimate=estimate,
                    )
                )

    best_case = min(
        feasible_cases,
        key=lambda case: case.estimate.takeoff_estimate.total_mass_kg,
        default=None,
    )
    return best_case, feasible_cases


def print_optimization_summary(best_case: BreguetOptimizationCase | None, feasible_case_count: int) -> None:
    print("")
    print("Breguet Input Optimizer")
    print(f"  Objective                = minimize takeoff mass")
    print(f"  Fuel volume limit        = {ALLOWED_FUEL_FRACTION:.2%} of vehicle volume")
    print(f"  Max thrust per engine    = {MAX_THRUST_PER_ENGINE_N:,.1f} N")
    print(f"  Thrust margin            = {THRUST_MARGIN:.2f}x")
    print(f"  Engine count options     = {ENGINE_COUNT_OPTIONS}")
    print(f"  Feasible cases           = {feasible_case_count}")

    if best_case is None:
        print("  No feasible case found with the current constraints.")
        return

    estimate = best_case.estimate
    engine_sizing = best_case.engine_sizing
    print("")
    print("Best Feasible Case")
    print(f"  Fuel                     = {best_case.fuel.name}")
    print(f"  Fuel density             = {best_case.fuel.density_kg_m3:,.1f} kg/m^3")
    print(f"  Specific impulse         = {best_case.specific_impulse_s:,.1f} s")
    print(f"  Engine count             = {best_case.engine_count}")
    print(f"  Available thrust         = {best_case.available_thrust_N:,.1f} N")
    print(f"  Required thrust + margin = {best_case.required_thrust_with_margin_N:,.1f} N")
    print(f"  Powerplant mass          = {estimate.powerplant_mass_kg:,.1f} kg")
    print(f"  Mass per engine          = {engine_sizing.powerplant_mass_per_engine_kg:,.1f} kg")
    print(f"  Required fuel mass       = {estimate.fuel_mass_kg:,.1f} kg")
    print(f"  Required fuel volume     = {best_case.fuel_volume_m3:,.1f} m^3")
    print(f"  Fuel volume fraction     = {best_case.fuel_fraction:.2%}")
    print(f"  X-51A fuel volume equiv. = {best_case.x51a_fuel_volume_equivalent:,.1f}x")
    print(f"  Takeoff mass             = {estimate.takeoff_estimate.total_mass_kg:,.1f} kg")
    print(f"  Takeoff weight           = {estimate.takeoff_estimate.total_weight_N:,.1f} N")
    print("")
    print("Best Case Mass Fractions")
    print(
        f"  Payload                  = "
        f"{mass_fraction(estimate.takeoff_estimate.payload_mass_kg, estimate.takeoff_estimate.total_mass_kg):.2%}"
    )
    print(
        f"  Airframe                 = "
        f"{mass_fraction(estimate.takeoff_estimate.airframe_mass_kg, estimate.takeoff_estimate.total_mass_kg):.2%}"
    )
    print(
        f"  Powerplant               = "
        f"{mass_fraction(estimate.takeoff_estimate.powerplant_mass_kg, estimate.takeoff_estimate.total_mass_kg):.2%}"
    )
    print(
        f"  Fuel                     = "
        f"{mass_fraction(estimate.takeoff_estimate.fuel_mass_kg, estimate.takeoff_estimate.total_mass_kg):.2%}"
    )
