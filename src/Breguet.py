"""Reusable Breguet-range utilities for preliminary fuel estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from weight import WeightEstimate, estimate_aircraft_weight


DEFAULT_RANGE_KM = 15900.0
DEFAULT_CRUISE_MACH = 6.0
DEFAULT_CRUISE_ALTITUDE_FT = 70000.0

FT_TO_M = 0.3048
GAMMA_AIR = 1.4
R_AIR_J_PER_KG_K = 287.05


@dataclass(frozen=True)
class BreguetRangeEstimate:
    volume_m3: float
    powerplant_mass_kg: float
    fuel_mass_kg: float
    lift_to_drag: float
    specific_impulse_s: float
    engine_count: int
    range_km: float
    cruise_mach: float
    cruise_altitude_ft: float
    temperature_k: float
    speed_of_sound_m_s: float
    cruise_speed_m_s: float
    time_of_flight_s: float
    mass_ratio: float
    zero_fuel_estimate: WeightEstimate
    takeoff_estimate: WeightEstimate


def isa_temperature_k(altitude_m: float) -> float:
    """Return ISA temperature at the requested geometric altitude."""
    if altitude_m < 0.0:
        raise ValueError("altitude_m must be non-negative.")

    if altitude_m <= 11000.0:
        return 288.15 - 0.0065 * altitude_m
    if altitude_m <= 20000.0:
        return 216.65
    if altitude_m <= 32000.0:
        return 216.65 + 0.001 * (altitude_m - 20000.0)

    raise ValueError("Altitude is outside the ISA layer range handled by this script.")


def speed_of_sound_m_s(temperature_k: float) -> float:
    """Return local speed of sound from static temperature."""
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive.")

    return math.sqrt(GAMMA_AIR * R_AIR_J_PER_KG_K * temperature_k)


def breguet_required_fuel_mass_kg(
    range_km: float,
    cruise_speed_m_s: float,
    lift_to_drag: float,
    specific_impulse_s: float,
    final_mass_kg: float,
) -> tuple[float, float]:
    """Return required fuel mass and the mass ratio Wi/Wf.

    Uses the jet Breguet form
        R = V * Isp * (L/D) * ln(W_i / W_f)
    with specific impulse supplied in seconds.
    """
    if range_km < 0.0:
        raise ValueError("range_km must be non-negative.")
    if cruise_speed_m_s <= 0.0:
        raise ValueError("cruise_speed_m_s must be positive.")
    if lift_to_drag <= 0.0:
        raise ValueError("lift_to_drag must be positive.")
    if specific_impulse_s <= 0.0:
        raise ValueError("specific_impulse_s must be positive.")
    if final_mass_kg <= 0.0:
        raise ValueError("final_mass_kg must be positive.")

    range_m = range_km * 1000.0
    log_mass_ratio = range_m / (cruise_speed_m_s * specific_impulse_s * lift_to_drag)
    mass_ratio = math.exp(log_mass_ratio)
    initial_mass_kg = final_mass_kg * mass_ratio
    fuel_mass_kg = initial_mass_kg - final_mass_kg
    return fuel_mass_kg, mass_ratio


def calculate_breguet_range_estimate(
    *,
    volume_m3: float,
    powerplant_mass_kg: float,
    lift_to_drag: float,
    specific_impulse_s: float,
    engine_count: int,
    range_km: float = DEFAULT_RANGE_KM,
    cruise_mach: float = DEFAULT_CRUISE_MACH,
    cruise_altitude_ft: float = DEFAULT_CRUISE_ALTITUDE_FT,
) -> BreguetRangeEstimate:
    """Estimate mission fuel mass and takeoff weight from the Breguet equation.

    The mission is flown at constant Mach and constant altitude. Weight
    calculations are delegated to ``weight.py``.
    """
    if engine_count <= 0:
        raise ValueError("engine_count must be positive.")

    zero_fuel_estimate = estimate_aircraft_weight(
        volume_m3=volume_m3,
        powerplant_mass_kg=powerplant_mass_kg,
        fuel_mass_kg=0.0,
    )

    cruise_altitude_m = cruise_altitude_ft * FT_TO_M
    temperature_k = isa_temperature_k(cruise_altitude_m)
    local_speed_of_sound_m_s = speed_of_sound_m_s(temperature_k)
    cruise_speed_m_s = cruise_mach * local_speed_of_sound_m_s
    time_of_flight_s = range_km * 1000.0 / cruise_speed_m_s

    fuel_mass_kg, mass_ratio = breguet_required_fuel_mass_kg(
        range_km=range_km,
        cruise_speed_m_s=cruise_speed_m_s,
        lift_to_drag=lift_to_drag,
        specific_impulse_s=specific_impulse_s,
        final_mass_kg=zero_fuel_estimate.zero_fuel_mass_kg,
    )

    takeoff_estimate = estimate_aircraft_weight(
        volume_m3=volume_m3,
        powerplant_mass_kg=powerplant_mass_kg,
        fuel_mass_kg=fuel_mass_kg,
    )

    return BreguetRangeEstimate(
        volume_m3=volume_m3,
        powerplant_mass_kg=powerplant_mass_kg,
        fuel_mass_kg=fuel_mass_kg,
        lift_to_drag=lift_to_drag,
        specific_impulse_s=specific_impulse_s,
        engine_count=engine_count,
        range_km=range_km,
        cruise_mach=cruise_mach,
        cruise_altitude_ft=cruise_altitude_ft,
        temperature_k=temperature_k,
        speed_of_sound_m_s=local_speed_of_sound_m_s,
        cruise_speed_m_s=cruise_speed_m_s,
        time_of_flight_s=time_of_flight_s,
        mass_ratio=mass_ratio,
        zero_fuel_estimate=zero_fuel_estimate,
        takeoff_estimate=takeoff_estimate,
    )
