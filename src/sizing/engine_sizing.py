"""Preliminary dry powerplant sizing utilities.

Fuel mass is intentionally not included here. Mission fuel should be estimated
by Breguet.py or another mission analysis module, then passed separately into
the aircraft weight model.
"""

from __future__ import annotations

from dataclasses import dataclass


STANDARD_GRAVITY_M_S2 = 9.80665

# Approximate engine thrust-to-weight ratio from A. Ingenito, S. Gulli, and
# C. Bruno, "Sizing of Scramjet Vehicles," Progress in Propulsion Physics 2
# (2011) 487-498, published by EDP Sciences, 2011.
# https://www.eucass-proceedings.eu/articles/eucass/pdf/2012/01/eucass2p487.pdf
ENGINE_THRUST_TO_WEIGHT_RATIO = 8.3


@dataclass(frozen=True)
class EngineSizingEstimate:
    required_thrust_N: float
    thrust_to_weight_ratio: float
    engine_count: int
    powerplant_mass_kg: float
    powerplant_mass_per_engine_kg: float
    powerplant_weight_N: float


def estimate_engine_sizing(
    *,
    required_thrust_N: float,
    engine_count: int = 1,
) -> EngineSizingEstimate:
    """Estimate dry powerplant mass from required thrust and engine T/W.

    Parameters
    ----------
    required_thrust_N
        Total installed thrust required by the vehicle.
    engine_count
        Number of engines sharing the required thrust.
    """
    if required_thrust_N < 0.0:
        raise ValueError("required_thrust_N must be non-negative.")
    if engine_count <= 0:
        raise ValueError("engine_count must be positive.")

    thrust_to_weight_ratio = ENGINE_THRUST_TO_WEIGHT_RATIO
    powerplant_weight_N = required_thrust_N / thrust_to_weight_ratio
    powerplant_mass_kg = powerplant_weight_N / STANDARD_GRAVITY_M_S2
    powerplant_mass_per_engine_kg = powerplant_mass_kg / engine_count

    return EngineSizingEstimate(
        required_thrust_N=required_thrust_N,
        thrust_to_weight_ratio=thrust_to_weight_ratio,
        engine_count=engine_count,
        powerplant_mass_kg=powerplant_mass_kg,
        powerplant_mass_per_engine_kg=powerplant_mass_per_engine_kg,
        powerplant_weight_N=powerplant_weight_N,
    )
