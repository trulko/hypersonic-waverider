Engine Sizing
=============

This folder now includes:

1. engine_sizing.py
2. Engine_Sizing_README.txt


Purpose
-------

engine_sizing.py estimates preliminary dry powerplant mass from:

- required thrust
- number of engines

It does not calculate or include mission fuel mass. Fuel is handled by
Breguet.py, so the same fuel is not counted twice.

The engine thrust-to-weight ratio is hard-coded in engine_sizing.py as:

    TWR_engine = 8.3

This value is cited in the code comments from:

    A. Ingenito, S. Gulli, and C. Bruno, "Sizing of Scramjet Vehicles,"
    Progress in Propulsion Physics 2 (2011) 487-498, published by
    EDP Sciences, 2011.
    https://www.eucass-proceedings.eu/articles/eucass/pdf/2012/01/eucass2p487.pdf


Main Equations
--------------

Powerplant weight:

    W_powerplant = F_required / TWR_engine

Powerplant mass:

    m_powerplant = W_powerplant / g0

Per-engine powerplant mass:

    m_per_engine = m_powerplant / N_engines

where:

- F_required is total required installed thrust, N
- TWR_engine is engine thrust-to-weight ratio, dimensionless
- g0 = 9.80665 m/s^2
- N_engines is the number of engines


Fuel Note
---------

Breguet.py provides the required fuel mass from the range equation. The
recommended flow is:

1. Use engine_sizing.py to estimate powerplant_mass_kg.
2. Pass powerplant_mass_kg into Breguet.py.
3. Let Breguet.py compute fuel_mass_kg.
4. Pass both powerplant_mass_kg and fuel_mass_kg into weight.py.


Example
-------

    from engine_sizing import estimate_engine_sizing

    estimate = estimate_engine_sizing(
        required_thrust_N=250000.0,
        engine_count=2,
    )

    print(estimate.powerplant_mass_kg)
    print(estimate.powerplant_mass_per_engine_kg)
