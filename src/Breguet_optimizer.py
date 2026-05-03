"""Discrete optimizer for fixed-geometry Breguet input assumptions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ViableOptionSummary:
    fuel: FuelOption
    specific_impulse_s: float
    engine_counts: tuple[int, ...]
    fuel_volume_m3: float
    fuel_fraction: float
    takeoff_mass_kg: float


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


def total_optimizer_cases() -> int:
    """Return the total number of discrete combinations in the sweep."""
    return sum(len(isp_samples(fuel)) * len(ENGINE_COUNT_OPTIONS) for fuel in FUEL_OPTIONS)


def summarize_viable_options(feasible_cases: list[BreguetOptimizationCase]) -> list[ViableOptionSummary]:
    """Collapse repeated feasible cases into unique fuel/Isp performance options."""
    grouped_cases: dict[tuple[str, float, float, float], list[BreguetOptimizationCase]] = {}

    for case in sorted(feasible_cases, key=lambda item: (item.fuel.name, item.specific_impulse_s, item.engine_count)):
        key = (
            case.fuel.name,
            round(case.specific_impulse_s, 6),
            round(case.fuel_volume_m3, 6),
            round(case.estimate.takeoff_estimate.total_mass_kg, 6),
        )
        grouped_cases.setdefault(key, []).append(case)

    option_summaries = []
    for grouped in grouped_cases.values():
        reference_case = grouped[0]
        option_summaries.append(
            ViableOptionSummary(
                fuel=reference_case.fuel,
                specific_impulse_s=reference_case.specific_impulse_s,
                engine_counts=tuple(case.engine_count for case in grouped),
                fuel_volume_m3=reference_case.fuel_volume_m3,
                fuel_fraction=reference_case.fuel_fraction,
                takeoff_mass_kg=reference_case.estimate.takeoff_estimate.total_mass_kg,
            )
        )

    return sorted(option_summaries, key=lambda item: (item.fuel.name, item.specific_impulse_s, item.engine_counts))


def save_feasible_cases_csv(feasible_cases: list[BreguetOptimizationCase], save_path: str | Path) -> Path:
    """Write the feasible optimizer cases to CSV for downstream analysis."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "fuel",
                "engine_count",
                "specific_impulse_s",
                "fuel_density_kg_m3",
                "fuel_mass_kg",
                "fuel_volume_m3",
                "fuel_volume_fraction",
                "x51a_fuel_volume_equivalent",
                "available_thrust_N",
                "required_thrust_with_margin_N",
                "powerplant_mass_kg",
                "powerplant_mass_per_engine_kg",
                "takeoff_mass_kg",
                "takeoff_weight_N",
            )
        )
        for case in sorted(feasible_cases, key=lambda item: (item.fuel.name, item.engine_count, item.specific_impulse_s)):
            writer.writerow(
                (
                    case.fuel.name,
                    case.engine_count,
                    f"{case.specific_impulse_s:.1f}",
                    f"{case.fuel.density_kg_m3:.1f}",
                    f"{case.estimate.fuel_mass_kg:.1f}",
                    f"{case.fuel_volume_m3:.3f}",
                    f"{case.fuel_fraction:.6f}",
                    f"{case.x51a_fuel_volume_equivalent:.3f}",
                    f"{case.available_thrust_N:.1f}",
                    f"{case.required_thrust_with_margin_N:.1f}",
                    f"{case.engine_sizing.powerplant_mass_kg:.1f}",
                    f"{case.engine_sizing.powerplant_mass_per_engine_kg:.1f}",
                    f"{case.estimate.takeoff_estimate.total_mass_kg:.1f}",
                    f"{case.estimate.takeoff_estimate.total_weight_N:.1f}",
                )
            )

    return output_path


def plot_feasible_cases(
    feasible_cases: list[BreguetOptimizationCase],
    best_case: BreguetOptimizationCase | None,
    save_path: str | Path,
) -> Path:
    """Save a fuel-by-fuel map of feasible optimizer cases."""
    import os

    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".mplconfig"))

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(FUEL_OPTIONS), figsize=(15.5, 4.8), constrained_layout=True)
    if len(FUEL_OPTIONS) == 1:
        axes = [axes]

    masses = [case.estimate.takeoff_estimate.total_mass_kg for case in feasible_cases]
    normalize = Normalize(vmin=min(masses), vmax=max(masses)) if masses else None
    scatter_artist = None

    for axis, fuel in zip(axes, FUEL_OPTIONS):
        fuel_cases = [
            case for case in feasible_cases if case.fuel.name == fuel.name
        ]

        axis.set_title(f"{fuel.name}\n{len(fuel_cases)} feasible case(s)")
        axis.set_xlabel("Engine count")
        axis.set_xticks(ENGINE_COUNT_OPTIONS)
        axis.set_xlim(min(ENGINE_COUNT_OPTIONS) - 0.8, max(ENGINE_COUNT_OPTIONS) + 0.8)
        axis.set_ylim(fuel.min_isp_s - 50.0, fuel.max_isp_s + 50.0)
        axis.grid(True, linestyle="--", alpha=0.35)

        if fuel_cases:
            scatter_artist = axis.scatter(
                [case.engine_count for case in fuel_cases],
                [case.specific_impulse_s for case in fuel_cases],
                c=[case.estimate.takeoff_estimate.total_mass_kg for case in fuel_cases],
                cmap="viridis_r",
                norm=normalize,
                s=140.0,
                edgecolors="#0b132b",
                linewidths=0.7,
                zorder=3,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No feasible\ncases",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#6c757d",
            )

        if best_case is not None and best_case.fuel.name == fuel.name:
            axis.scatter(
                best_case.engine_count,
                best_case.specific_impulse_s,
                marker="*",
                s=380.0,
                facecolors="none",
                edgecolors="#c1121f",
                linewidths=2.0,
                zorder=4,
            )
            axis.annotate(
                "Best",
                xy=(best_case.engine_count, best_case.specific_impulse_s),
                xytext=(10, -18),
                textcoords="offset points",
                color="#c1121f",
                fontsize=10,
                fontweight="bold",
            )

    axes[0].set_ylabel("Specific impulse, $I_{sp}$ [s]")
    figure.suptitle(
        "Breguet optimizer viable design space",
        fontsize=14,
        fontweight="bold",
    )

    if scatter_artist is not None:
        colorbar = figure.colorbar(scatter_artist, ax=axes, shrink=0.86, pad=0.02)
        colorbar.set_label("Takeoff mass [kg]")

    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def latex_escape(text: str) -> str:
    """Escape a string for safe inclusion in simple LaTeX text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped_text = text
    for source, replacement in replacements.items():
        escaped_text = escaped_text.replace(source, replacement)
    return escaped_text


def build_latex_summary(
    best_case: BreguetOptimizationCase | None,
    feasible_cases: list[BreguetOptimizationCase],
    *,
    volume_m3: float,
    lift_to_drag: float,
    required_thrust_N: float,
    plot_include_path: str,
) -> str:
    """Return a LaTeX-ready summary of the optimizer results."""
    total_cases = total_optimizer_cases()
    feasible_fraction_percent = 100.0 * len(feasible_cases) / total_cases if total_cases else 0.0
    viable_fuels = sorted({case.fuel.name for case in feasible_cases})
    viable_fuel_text = ", ".join(latex_escape(fuel_name) for fuel_name in viable_fuels) if viable_fuels else "none"
    unique_options = summarize_viable_options(feasible_cases)

    lines = [
        r"\subsection{Breguet Optimizer Trade Study}",
        (
            "The discrete Breguet optimizer swept fuel type, engine count, and specific impulse "
            f"for the fixed-geometry baseline ($V={volume_m3:.1f}\\,\\mathrm{{m^3}}$, "
            f"$L/D={lift_to_drag:.4f}$, and $T_\\mathrm{{req}}={required_thrust_N:,.1f}\\,\\mathrm{{N}}$). "
            f"The sweep covered {total_cases} total combinations with a fuel-volume cap of "
            f"{ALLOWED_FUEL_FRACTION * 100.0:.0f}\\%, a thrust margin of {THRUST_MARGIN:.2f}, and a per-engine "
            f"thrust limit of {MAX_THRUST_PER_ENGINE_N:,.0f}\\,\\mathrm{{N}}."
        ),
        "",
        (
            f"Out of these {total_cases} combinations, {len(feasible_cases)} cases "
            f"({feasible_fraction_percent:.1f}\\%) were feasible. "
            f"The viable set was limited to {viable_fuel_text} under the current mission and geometry assumptions."
        ),
        "",
        r"\begin{figure}[htbp]",
        r"    \centering",
        f"    \\includegraphics[width=0.98\\linewidth]{{{plot_include_path}}}",
        (
            r"    \caption{Viable Breguet optimizer cases by fuel option. Marker color denotes takeoff mass, "
            r"and the red star marks the minimum-mass feasible case.}"
        ),
        r"    \label{fig:breguet-optimizer-viable-options}",
        r"\end{figure}",
        "",
    ]

    if unique_options:
        lines.extend(
            [
                r"\begin{table}[htbp]",
                r"    \centering",
                r"    \caption{Unique viable Breguet optimizer options for the baseline configuration.}",
                r"    \begin{tabular}{lcccc}",
                r"        \hline",
                r"        Fuel & $I_{sp}$ [s] & Feasible engine counts & Fuel volume fraction & Takeoff mass [kg] \\",
                r"        \hline",
            ]
        )

        for option in unique_options:
            engine_count_text = ", ".join(str(engine_count) for engine_count in option.engine_counts)
            lines.append(
                "        "
                f"{latex_escape(option.fuel.name)} & "
                f"{option.specific_impulse_s:.0f} & "
                f"{engine_count_text} & "
                f"{option.fuel_fraction * 100.0:.2f}\\% & "
                f"{option.takeoff_mass_kg:,.1f} \\\\"
            )

        lines.extend(
            [
                r"        \hline",
                r"    \end{tabular}",
                r"    \label{tab:breguet-optimizer-viable-options}",
                r"\end{table}",
                "",
            ]
        )

    if best_case is None:
        lines.append(
            "No feasible case satisfied the combined thrust and fuel-volume constraints, so the next design step "
            "would be to relax the mission requirements or change the baseline geometry."
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            (
                f"The minimum-mass feasible case used {latex_escape(best_case.fuel.name)} with "
                f"$I_{{sp}}={best_case.specific_impulse_s:.0f}\\,\\mathrm{{s}}$ and {best_case.engine_count} engines. "
                f"This case required {best_case.fuel_volume_m3:,.1f}\\,\\mathrm{{m^3}} of fuel "
                f"({best_case.fuel_fraction * 100.0:.2f}\\% of vehicle volume) and produced a takeoff mass of "
                f"{best_case.estimate.takeoff_estimate.total_mass_kg:,.1f}\\,\\mathrm{{kg}}."
            ),
            "",
            (
                "Because the current engine-sizing model keeps total installed powerplant mass fixed for a fixed "
                "thrust requirement and thrust-to-weight ratio, engine count affects feasibility through per-engine "
                "thrust capacity but does not change takeoff mass once the thrust constraint is met. "
                f"As a result, the first feasible engine count ({best_case.engine_count}) is also tied with all "
                "larger feasible engine counts at the same specific impulse."
            ),
            "",
            (
                "Within the feasible design space, increasing specific impulse monotonically reduced both fuel-volume "
                "fraction and takeoff mass. This indicates that propulsion efficiency, rather than additional engine "
                "multiplicity, is the dominant lever in the present Breguet-based trade study."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def save_latex_summary(latex_summary: str, save_path: str | Path) -> Path:
    """Write a LaTeX-ready optimizer summary to disk."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_summary, encoding="utf-8")
    return output_path
