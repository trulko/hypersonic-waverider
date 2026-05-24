"""Simple aircraft weight estimation from enclosed vehicle volume.

Current model assumptions:
- 100 passengers
- 82.2 kg per passenger
- 16.0 kg of luggage per passenger
- airframe mass density estimated from 737-200 empty structure data

Fuel mass is left as an optional input and defaults to zero until a propulsion
and mission fuel model is added.
"""

from __future__ import annotations

from dataclasses import dataclass


PASSENGER_COUNT = 100
PASSENGER_MASS_KG = 82.2
LUGGAGE_MASS_KG = 16.0

# 737-200 reference:
# (54,249 - 8,177) kg / 750 m^3 = 61.43 kg/m^3
AIRFRAME_MASS_PER_VOLUME_KG_M3 = (54249.0 - 8177.0) / 750.0
STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class WeightEstimate:
    volume_m3: float
    passenger_count: int
    payload_mass_kg: float
    airframe_mass_kg: float
    powerplant_mass_kg: float
    fuel_mass_kg: float
    zero_fuel_mass_kg: float
    total_mass_kg: float
    zero_fuel_weight_N: float
    total_weight_N: float


def estimate_payload_mass_kg(passenger_count: int = PASSENGER_COUNT) -> float:
    """Return passenger + luggage mass."""
    if passenger_count < 0:
        raise ValueError("passenger_count must be non-negative.")

    return passenger_count * (PASSENGER_MASS_KG + LUGGAGE_MASS_KG)


def estimate_aircraft_weight(
    volume_m3: float,
    powerplant_mass_kg: float,
    fuel_mass_kg: float = 0.0,
    passenger_count: int = 100,
    airframe_mass_per_volume_kg_m3: float = AIRFRAME_MASS_PER_VOLUME_KG_M3,
) -> WeightEstimate:
    """Estimate aircraft mass and weight from vehicle volume.

    Parameters
    ----------
    volume_m3
        Enclosed vehicle volume in cubic meters.
    powerplant_mass_kg
        Installed powerplant mass.
    fuel_mass_kg
        Fuel carried for the mission. Defaults to 0 kg for now.
    passenger_count
        Number of passengers. Defaults to 100.
    airframe_mass_per_volume_kg_m3
        Structural mass-density estimate for the vehicle.
    """
    if volume_m3 < 0.0:
        raise ValueError("volume_m3 must be non-negative.")
    if fuel_mass_kg < 0.0:
        raise ValueError("fuel_mass_kg must be non-negative.")
    if airframe_mass_per_volume_kg_m3 < 0.0:
        raise ValueError("airframe_mass_per_volume_kg_m3 must be non-negative.")
    if powerplant_mass_kg < 0.0:
        raise ValueError("powerplant_mass_kg must be non-negative.")

    payload_mass_kg = estimate_payload_mass_kg(passenger_count)
    airframe_mass_kg = airframe_mass_per_volume_kg_m3 * volume_m3

    zero_fuel_mass_kg = payload_mass_kg + airframe_mass_kg + powerplant_mass_kg
    total_mass_kg = zero_fuel_mass_kg + fuel_mass_kg

    return WeightEstimate(
        volume_m3=volume_m3,
        passenger_count=passenger_count,
        payload_mass_kg=payload_mass_kg,
        airframe_mass_kg=airframe_mass_kg,
        powerplant_mass_kg=powerplant_mass_kg,
        fuel_mass_kg=fuel_mass_kg,
        zero_fuel_mass_kg=zero_fuel_mass_kg,
        total_mass_kg=total_mass_kg,
        zero_fuel_weight_N=zero_fuel_mass_kg * STANDARD_GRAVITY_M_S2,
        total_weight_N=total_mass_kg * STANDARD_GRAVITY_M_S2,
    )
