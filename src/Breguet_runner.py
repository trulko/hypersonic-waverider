"""Run the reusable Breguet-range function with the current project assumptions."""

from __future__ import annotations

from Breguet import calculate_breguet_range_estimate


ASSUMED_VOLUME_M3 = 750.0
ASSUMED_POWERPLANT_MASS_KG = 8500.0
ASSUMED_L_OVER_D = 3.7879
ASSUMED_ISP_S = 1900.0
ASSUMED_ENGINE_COUNT = 2


def main() -> None:
    estimate = calculate_breguet_range_estimate(
        volume_m3=ASSUMED_VOLUME_M3,
        powerplant_mass_kg=ASSUMED_POWERPLANT_MASS_KG,
        lift_to_drag=ASSUMED_L_OVER_D,
        specific_impulse_s=ASSUMED_ISP_S,
        engine_count=ASSUMED_ENGINE_COUNT,
    )

    print("Breguet Range Estimate")
    print(f"  Range                     = {estimate.range_km:,.0f} km")
    print(f"  Cruise altitude           = {estimate.cruise_altitude_ft:,.0f} ft")
    print(f"  ISA temperature           = {estimate.temperature_k:.2f} K")
    print(f"  Cruise Mach number        = {estimate.cruise_mach:.1f}")
    print(f"  Cruise speed              = {estimate.cruise_speed_m_s:,.1f} m/s")
    print(f"  Time of flight            = {estimate.time_of_flight_s/3600.0:.2f} hr")
    print(f"  Lift-to-drag ratio        = {estimate.lift_to_drag:.4f}")
    print(f"  Specific impulse          = {estimate.specific_impulse_s:,.1f} s")
    print(f"  Number of engines         = {estimate.engine_count}")
    print(f"  Mass ratio Wi/Wf          = {estimate.mass_ratio:.4f}")
    print("")
    print("Weight Breakdown")
    print(f"  Payload mass             = {estimate.takeoff_estimate.payload_mass_kg:,.1f} kg")
    print(f"  Airframe mass            = {estimate.takeoff_estimate.airframe_mass_kg:,.1f} kg")
    print(f"  Powerplant mass          = {estimate.takeoff_estimate.powerplant_mass_kg:,.1f} kg")
    print(f"  Required fuel mass       = {estimate.takeoff_estimate.fuel_mass_kg:,.1f} kg")
    print(f"  Zero-fuel mass           = {estimate.takeoff_estimate.zero_fuel_mass_kg:,.1f} kg")
    print(f"  Takeoff mass             = {estimate.takeoff_estimate.total_mass_kg:,.1f} kg")
    print(f"  Takeoff weight           = {estimate.takeoff_estimate.total_weight_N:,.1f} N")
    print("")
    print("Notes")
    print("  The Breguet calculation uses constant Mach, constant L/D, and constant Isp.")
    print("  Engine count is recorded as an assumption, but it does not change the result in this formulation.")
    print("  No reserve, climb, acceleration, or descent fuel is included.")


if __name__ == "__main__":
    main()
