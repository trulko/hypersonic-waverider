"""Run the reusable Breguet-range function with the current project assumptions."""

from __future__ import annotations

from Breguet import calculate_breguet_range_estimate
from Breguet_optimizer import (
    X51A_FUEL_DENSITY_KG_M3,
    X51A_FUEL_VOLUME_M3,
    estimate_fuel_storage_volume_m3,
    optimize_breguet_inputs,
    print_optimization_summary,
)
from engine_sizing import estimate_engine_sizing
from main import build_waverider
from Thruster_I_Hardly_Even_Know_Her import Thruster_I_Hardly_Even_Know_Her


HARDCODED_REQUIRED_THRUST_N = 468393.5
HARDCODED_L_OVER_D = 4.4656
ASSUMED_ISP_S = 1900.0
ASSUMED_ENGINE_COUNT = 2
L_OVER_D_SOURCE = "main"
THRUST_SOURCE = "thruster"
RUN_OPTIMIZER = True


def get_volume(waverider) -> tuple[float, str]:
    """Return vehicle volume and a short label describing where it came from."""
    return float(waverider.panel.volume), "main.py"


def get_lift_to_drag(waverider) -> tuple[float, str]:
    """Return L/D and a short label describing where it came from."""
    if L_OVER_D_SOURCE == "hardcoded":
        return HARDCODED_L_OVER_D, "hardcoded"
    if L_OVER_D_SOURCE == "main":
        if waverider.LD_total is None:
            raise RuntimeError("main.py did not produce a total L/D value.")
        return waverider.LD_total, "main.py"
    raise ValueError("L_OVER_D_SOURCE must be either 'main' or 'hardcoded'.")


def get_required_thrust() -> tuple[float, str]:
    """Return required thrust and a short label describing where it came from."""
    if THRUST_SOURCE == "hardcoded":
        return HARDCODED_REQUIRED_THRUST_N, "hardcoded"
    if THRUST_SOURCE == "thruster":
        return Thruster_I_Hardly_Even_Know_Her().required_thrust_N, "Thruster_I_Hardly_Even_Know_Her.py"
    raise ValueError("THRUST_SOURCE must be either 'hardcoded' or 'thruster'.")


def mass_fraction(component_mass_kg: float, total_mass_kg: float) -> float:
    """Return component mass as a fraction of total mass."""
    if total_mass_kg <= 0.0:
        raise ValueError("total_mass_kg must be positive.")

    return component_mass_kg / total_mass_kg


def main() -> None:
    waverider = build_waverider()
    volume_m3, volume_source = get_volume(waverider)
    lift_to_drag, lift_to_drag_source = get_lift_to_drag(waverider)
    required_thrust_N, required_thrust_source = get_required_thrust()

    engine_sizing = estimate_engine_sizing(
        required_thrust_N=required_thrust_N,
        engine_count=ASSUMED_ENGINE_COUNT,
    )

    estimate = calculate_breguet_range_estimate(
        volume_m3=volume_m3,
        powerplant_mass_kg=engine_sizing.powerplant_mass_kg,
        lift_to_drag=lift_to_drag,
        specific_impulse_s=ASSUMED_ISP_S,
        engine_count=ASSUMED_ENGINE_COUNT,
    )
    fuel_storage_volume_m3 = estimate_fuel_storage_volume_m3(estimate.fuel_mass_kg)
    x51a_fuel_volume_equivalent = fuel_storage_volume_m3 / X51A_FUEL_VOLUME_M3

    print("Breguet Range Estimate")
    print(f"  Range                     = {estimate.range_km:,.0f} km")
    print(f"  Cruise altitude           = {estimate.cruise_altitude_ft:,.0f} ft")
    print(f"  ISA temperature           = {estimate.temperature_k:.2f} K")
    print(f"  Cruise Mach number        = {estimate.cruise_mach:.1f}")
    print(f"  Cruise speed              = {estimate.cruise_speed_m_s:,.1f} m/s")
    print(f"  Time of flight            = {estimate.time_of_flight_s/3600.0:.2f} hr")
    print(f"  Vehicle volume            = {estimate.volume_m3:,.3f} m^3")
    print(f"  Volume source             = {volume_source}")
    print(f"  Lift-to-drag ratio        = {estimate.lift_to_drag:.4f}")
    print(f"  L/D source                = {lift_to_drag_source}")
    print(f"  Specific impulse          = {estimate.specific_impulse_s:,.1f} s")
    print(f"  Number of engines         = {estimate.engine_count}")
    print(f"  Required thrust           = {engine_sizing.required_thrust_N:,.1f} N")
    print(f"  Required thrust/engine    = {engine_sizing.required_thrust_N / estimate.engine_count:,.1f} N")
    print(f"  Required thrust source    = {required_thrust_source}")
    print(f"  Engine thrust/weight      = {engine_sizing.thrust_to_weight_ratio:.2f}")
    print(f"  Mass ratio Wi/Wf          = {estimate.mass_ratio:.4f}")
    print("")
    print("Weight Breakdown")
    print(f"  Payload mass             = {estimate.takeoff_estimate.payload_mass_kg:,.1f} kg")
    print(f"  Airframe mass            = {estimate.takeoff_estimate.airframe_mass_kg:,.1f} kg")
    print(f"  Powerplant mass          = {estimate.takeoff_estimate.powerplant_mass_kg:,.1f} kg")
    print(f"  Mass per engine          = {engine_sizing.powerplant_mass_per_engine_kg:,.1f} kg")
    print(f"  Required fuel mass       = {estimate.takeoff_estimate.fuel_mass_kg:,.1f} kg")
    print(f"  JP-7 fuel density        = {X51A_FUEL_DENSITY_KG_M3:,.1f} kg/m^3")
    print(f"  Required fuel volume     = {fuel_storage_volume_m3:,.1f} m^3")
    print(f"  X-51A fuel volume equiv. = {x51a_fuel_volume_equivalent:,.1f}x")
    print(f"  Zero-fuel mass           = {estimate.takeoff_estimate.zero_fuel_mass_kg:,.1f} kg")
    print(f"  Takeoff mass             = {estimate.takeoff_estimate.total_mass_kg:,.1f} kg")
    print(f"  Takeoff weight           = {estimate.takeoff_estimate.total_weight_N:,.1f} N")
    print("")
    print("Mass Fractions")
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
    print("")
    print("Notes")
    print("  The Breguet calculation uses constant Mach, constant L/D, and constant Isp.")
    print("  Powerplant mass is computed by engine_sizing.py before fuel is estimated.")
    print("  No reserve, climb, acceleration, or descent fuel is included.")

    if RUN_OPTIMIZER:
        best_case, feasible_cases = optimize_breguet_inputs(
            volume_m3=volume_m3,
            lift_to_drag=lift_to_drag,
            required_thrust_N=required_thrust_N,
        )
        print_optimization_summary(best_case, len(feasible_cases))


if __name__ == "__main__":
    main()
