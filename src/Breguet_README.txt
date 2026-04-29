Breguet Scripts
================

This folder contains two related scripts:

1. Breguet.py
2. Breguet_runner.py


1. Breguet.py
-------------

Purpose:
This is the reusable calculation module. It contains the functions needed to
estimate fuel mass and aircraft takeoff weight using the jet Breguet range
equation.

What it does:
- Computes ISA temperature at the cruise altitude
- Computes local speed of sound
- Computes cruise speed from Mach number
- Computes time of flight from range and cruise speed
- Uses the Breguet range equation to estimate required fuel mass
- Uses weight.py to compute zero-fuel and takeoff aircraft weight

Main function:
- calculate_breguet_range_estimate(...)

Important note:
- Breguet.py does NOT assume powerplant mass or specific impulse internally.
  Those must be supplied by the caller.

Default mission settings inside Breguet.py:
- Range = 15,900 km
- Cruise Mach number = 6.0
- Cruise altitude = 70,000 ft


2. Breguet_runner.py
--------------------

Purpose:
This is the user-facing driver script. It calls the reusable function in
Breguet.py with the current project assumptions and prints a formatted summary
to the terminal.

Current assumptions used in Breguet_runner.py:
- Vehicle volume = 750 m^3
- Powerplant mass = 8,500 kg
- Lift-to-drag ratio, L/D = 3.7879
- Specific impulse, Isp = 1900 s
- Number of engines = 2

What it prints:
- Range
- Cruise altitude
- ISA temperature
- Cruise Mach number
- Cruise speed
- Time of flight
- Lift-to-drag ratio
- Specific impulse
- Number of engines
- Mass ratio Wi/Wf
- Payload mass
- Airframe mass
- Powerplant mass
- Required fuel mass
- Zero-fuel mass
- Takeoff mass
- Takeoff weight


Assumptions in the Breguet Method
---------------------------------

The Breguet estimate in these scripts assumes:
- Constant Mach number throughout cruise
- Constant cruise altitude of 70,000 ft
- Constant lift-to-drag ratio
- Constant specific impulse
- No climb fuel
- No acceleration fuel
- No descent fuel
- No reserve fuel
- No loiter segment

Because of these assumptions, the result should be treated as a first-order
conceptual estimate, not a full mission fuel analysis.


Connection to weight.py
-----------------------

Both scripts rely on weight.py to estimate the aircraft mass from:
- enclosed vehicle volume
- payload assumptions
- airframe mass-per-volume estimate
- user-supplied powerplant mass
- computed fuel mass

This means the Breguet fuel estimate and the aircraft weight estimate are
coupled consistently.


How to Use
----------

To use the reusable function in another script:

    from Breguet import calculate_breguet_range_estimate

    estimate = calculate_breguet_range_estimate(
        volume_m3=750.0,
        powerplant_mass_kg=8500.0,
        lift_to_drag=3.7879,
        specific_impulse_s=1900.0,
        engine_count=2,
    )

To run the assumption-based script directly from src:

    ..\.venv\Scripts\python.exe Breguet_runner.py
