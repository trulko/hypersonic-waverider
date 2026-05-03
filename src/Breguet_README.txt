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
- Vehicle volume = from main.py baseline waverider geometry
- Required thrust = 468,393.5 N when THRUST_SOURCE = "hardcoded"
- Required thrust can be recomputed from Thruster_I_Hardly_Even_Know_Her.py
  when THRUST_SOURCE = "thruster"
- Engine thrust-to-weight ratio = 8.3 from engine_sizing.py
- Lift-to-drag ratio, L/D = from main.py when L_OVER_D_SOURCE = "main"
- Specific impulse, Isp = 1900 s
- Number of engines = 2
- Fuel storage volume uses JP-7 density = 803 kg/m^3, using the X-51A
  JP-7 fuel choice as the reference.
- If RUN_OPTIMIZER = True, Breguet_optimizer.py sweeps fuel type, engine count,
  and specific impulse to minimize takeoff mass with fixed geometry.

Optimizer settings in Breguet_optimizer.py:
- ALLOWED_FUEL_FRACTION = 0.15
- THRUST_MARGIN = 1.20
- MAX_THRUST_PER_ENGINE_N = 60,000 N
  Source:
  https://www.afrl.af.mil/News/Article-Display/Article/2339336/afrl-achieves-record-setting-hypersonic-ground-test-milestone/
- ENGINE_COUNT_OPTIONS = (2, 4, 6, 8, 10, 12, 14, 16)
- FUEL_OPTIONS = JP-7, liquid methane, liquid hydrogen

What it prints:
- Range
- Cruise altitude
- ISA temperature
- Cruise Mach number
- Cruise speed
- Time of flight
- Vehicle volume
- Volume source
- Lift-to-drag ratio
- Specific impulse
- Number of engines
- Required thrust
- Required thrust per engine
- Required thrust source
- Engine thrust-to-weight ratio
- Mass ratio Wi/Wf
- Payload mass
- Airframe mass
- Powerplant mass
- Mass per engine
- Required fuel mass
- JP-7 fuel density
- Required fuel volume
- X-51A fuel volume equivalent
- Zero-fuel mass
- Takeoff mass
- Takeoff weight
- Payload, airframe, powerplant, and fuel mass fractions
- Optimizer objective and constraints
- Best feasible optimizer case, when RUN_OPTIMIZER = True

What it saves when RUN_OPTIMIZER = True:
- runs/breguet-optimizer/viable-options.png
  fuel-by-fuel design-space plot with viable cases and non-viable cases marked
- runs/breguet-optimizer/viable-options.csv
  raw list of all feasible discrete optimizer cases
- runs/breguet-optimizer/optimization-summary.tex
  LaTeX-ready subsection with a figure include, viable-options table, and
  summary paragraphs
- runs/breguet-optimizer/overleaf/
  Overleaf-ready bundle containing main.tex, optimization-summary.tex,
  viable-options.png, and a short upload README


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

Breguet_runner.py first uses engine_sizing.py to estimate dry powerplant mass
from required thrust and engine thrust-to-weight ratio. Then Breguet.py and
weight.py estimate the aircraft mass from:
- enclosed vehicle volume
- payload assumptions
- airframe mass-per-volume estimate
- powerplant mass from engine_sizing.py
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

To choose whether thrust is fast/hardcoded or fully recomputed, edit this
setting in Breguet_runner.py:

    THRUST_SOURCE = "hardcoded"

or:

    THRUST_SOURCE = "thruster"
