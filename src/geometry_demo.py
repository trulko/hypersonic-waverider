"""
Generate a small sweep of waverider geometries and plot a wireframe grid.

Run from src/:
    python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from waverider import Waverider
from pyvista_writer import plot_geometry_grid_pv


def main() -> None:
    cases = [
        {"R1_frac": 0.20, "W2_frac": 0.60, "n_shape": 0.85},
        {"R1_frac": 0.40, "W2_frac": 0.70, "n_shape": 1.15},
        {"R1_frac": 0.77, "W2_frac": 0.55, "n_shape": 1.75},
        {"R1_frac": 0.50, "W2_frac": 0.80, "n_shape": 1.05},
        {"R1_frac": 0.2, "W2_frac": 0.85, "n_shape": 0.75},
        {"R1_frac": 0.2, "W2_frac": 0.85, "n_shape": 1.1},
    ]

    meshes = []
    labels = []

    for idx, params in enumerate(cases, start=1):
        wv = Waverider(
            M1 = 6.0,
            gamma = 1.4,
            beta = 17,
            min_height = 3,   # [m]
            min_area = 100,   # [m^2]
            min_volume = 250, # [m^3]
            N = 300,
            N_l = 25,
            R1_frac = params["R1_frac"],
            W2_frac = params["W2_frac"],
            n_shape = params["n_shape"],
        )

        ld = wv.inviscid_aerodynamics()
        vol = wv.panel.volume
        area = wv.panel.wetted_area

        meshes.append((wv.panel.lower_mesh, wv.panel.upper_mesh))
        labels.append(
            "R1={r1:.2f}, W2={w2:.2f}, n={n:.2f}\n"
            "V={v:.2f} m^3, A={a:.2f} m^2, L/D={ld:.2f}".format(
                r1=params["R1_frac"],
                w2=params["W2_frac"],
                n=params["n_shape"],
                v=vol,
                a=area,
                ld=ld,
            )
        )

        print(
            "Case {i}: R1={r1:.2f}, W2={w2:.2f}, n={n:.2f} -> "
            "V={v:.2f} m^3, A={a:.2f} m^2, L/D={ld:.2f}".format(
                i=idx,
                r1=params["R1_frac"],
                w2=params["W2_frac"],
                n=params["n_shape"],
                v=vol,
                a=area,
                ld=ld,
            )
        )

    output_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "te_sweep")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "geometry_grid.png")

    height = max(600, 300 * len(meshes))
    plot_geometry_grid_pv(
        meshes,
        labels,
        window_size=(1100, height),
        save_path=out_path,
    )
    print("Saved plot to {path}".format(path=out_path))

if __name__ == "__main__":
    main()
