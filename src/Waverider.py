"""
Waverider design tool.
"""

import os
import numpy as np

from oblique_shock import Oblique_Shock
from taylor_maccoll_sol import Taylor_Maccoll
from TE_Formation import TEG
from streamline_tracing import TRACE
from mesh_panelization import Panelization, plot_scalar_field
from pyvista_writer import (
    plot_scalar_field_pv,
    plot_flowfield_slices_pv,
    plot_geometry_views_pv,
)
from aerodynamics import compute_inviscid_forces, compute_pressure
from boundary_layer import compute_skin_friction, skin_friction_on_mesh, getMaxWallShearStress
from blunting_correction import (
    minimum_blunting_radius,
    blunt_leading_edge_force,
    equilibrium_wall_temperature,
)

R_AIR = 287.05  # J/(kg K), specific gas constant for air

class Waverider:
    """A waverider geometry plus all aerothermodynamic post-processing.

    Constructing an instance runs the geometry pipeline (oblique shock,
    Taylor-Maccoll, trailing-edge curve, leading-edge projection, streamline
    tracing) and panelises the result.

    Parameters
    ----------
    M1, gamma, beta : freestream Mach, ratio of specific heats, shock
                      semi-angle (deg).
    L               : vehicle length (m).
    N               : leading-edge resolution.
    N_l             : number of lower-surface streamlines.
    N_up            : number of upper-surface lines.
    R1_frac, W2_frac, n_shape : trailing-edge curve parameters.

    Class attributes (after construction)
    -------------------------------------
    geometry  : the geometry dict (same content as the old design_waverider).
    teg       : the trailing-edge generator object.
    tracer    : the streamline tracer object.
    panel     : ``Panelization`` instance with lower/upper triangle meshes.

    Attributes set by ``aerothermodynamics(...)``
    ---------------------------------------------
    T_inf, p_inf, T_allow : freestream and design-allowable wall temp.
    rho_inf, V_inf        : freestream density (kg/m^3), velocity (m/s).
    pressure              : pressure dict from ``compute_pressure``.
    inviscid_forces       : ``compute_inviscid_forces`` result.
    viscous_forces        : ``compute_skin_friction`` result.
    blunt_forces          : ``blunt_leading_edge_force`` result.
    le_size               : ``minimum_blunting_radius`` result.
    R_n                   : minimum allowable LE bluntness (m).
    T_stag_check          : equilibrium stag-point wall T at R_n (K).
    CL_total, CD_total, LD_total : combined coefficients.
    """

    def __init__(self,
                 M1: float,
                 gamma: float,
                 beta: float,
                 min_volume: float,
                 min_height: float,
                 min_area: float,
                 N: int = 500,
                 N_l: int = 50,
                 R1_frac: float = 0.2,
                 W2_frac: float = 0.8,
                 n_shape: float = 1.0):
        self.M1      = M1
        self.gamma   = gamma
        self.beta    = beta
        self.N       = N
        self.N_l     = N_l
        self.N_up    = N_l
        self.R1_frac = R1_frac
        self.W2_frac = W2_frac
        self.n_shape = n_shape

        self.L        = self._get_minimum_length(min_volume, min_area, min_height)
        self.geometry = self._build_geometry(self.L)
        self.panel    = Panelization(self.geometry)
        self.vehicle_length = self.tracer.vehicle_length(self.geometry)

        # aerothermo results are filled by aerothermodynamics(); set defaults.
        self.T_inf = self.p_inf = self.T_allow = None
        self.rho_inf = self.V_inf = None
        self.pressure = None
        self.inviscid_forces = None
        self.viscous_forces  = None
        self.blunt_forces    = None
        self.le_size         = None
        self.R_n             = None
        self.T_stag_check    = None
        self.CL_total = self.CD_total = self.LD_total = None

    # -----------------------------------------------------------
    def _get_minimum_length(self, target_volume, target_area, target_height) -> float:
        """
        Determines the required length L to meet minimum volume, area, and height constraints.
        """
        # 1. Instantiate a "Unit" Waverider (L = 1.0)
        unit_geometry = self._build_geometry(1.0)
        unit_panel = Panelization(unit_geometry)
        unit_volume = unit_panel.volume
        unit_height = unit_panel.height
        unit_area = unit_panel.wetted_area

        # 2. Apply Scaling Laws
        L_req_vol = (target_volume / unit_volume) ** (1.0 / 3.0)
        L_req_area = (target_area / unit_area) ** (1.0 / 2.0)
        L_req_height = (target_height / unit_height)
        L_final = max(L_req_vol, L_req_height, L_req_area)

        return L_final

    # -----------------------------------------------------------
    def _build_geometry(self, L : float) -> dict:
        # Build backface geometry
        self.teg = TEG(self.gamma)
        z_func, Rs = self.teg.make_simple_backface(
            L=L, beta_deg=self.beta,
            R1_frac=self.R1_frac, W_frac=self.W2_frac,
            n_shape=self.n_shape,
        )
        self._z_func = z_func
        self._Rs     = Rs
        beta_rad = np.radians(self.beta)

        # Oblique shock initial conditions
        os_solver = Oblique_Shock()
        M2, theta_deg, beta_rad_out, theta_rad = os_solver.sub_1(
            self.M1, self.gamma, self.beta)
        Vr_i, V_theta_i = os_solver.initial_nondimensioned_conditions(
            self.M1, self.gamma, self.beta)

        # Cone half-angle from Taylor-Maccoll
        tm = Taylor_Maccoll(self.gamma)
        cone_angle = tm.cone_half_angle(beta_rad, Vr_i, V_theta_i)

        # Streamline tracing
        self.tracer = TRACE(self.gamma)
        geometry = self.tracer.tracing_module(
            z_func, Rs, L, self.N, self.N_l, self.N_up, Vr_i, V_theta_i)

        # Cache projection used by plot_geometry
        _, _, _, X_p, Y_p, Z_p, *_ = self.tracer.projection_module(
            z_func, Rs, L, self.N)
        self._X_p, self._Y_p, self._Z_p = X_p, Y_p, Z_p

        geometry["shock_conditions"] = {
            "M2":         float(M2),
            "theta_deg":  float(theta_deg),
            "beta_rad":   float(beta_rad_out),
            "theta_rad":  float(theta_rad),
            "Vr_i":       float(Vr_i),
            "V_theta_i":  float(V_theta_i),
            "cone_half_angle_deg": float(np.degrees(cone_angle)),
        }
        geometry["parameters"] = {
            "M1": self.M1, "gamma": self.gamma, "beta": self.beta,
            "L": L, "N": self.N, "N_l": self.N_l, "N_up": self.N_up,
            "Rs": Rs,
        }
        return geometry

    # -----------------------------------------------------------
    def inviscid_aerodynamics(self) -> float:
        """Compute inviscid aerodynamic forces and return inviscid L/D."""
        self.pressure        = compute_pressure(self.geometry, self.panel.lower_mesh)
        self.inviscid_forces = compute_inviscid_forces(self.geometry, self.panel.lower_mesh)
        return self.inviscid_forces["L_over_D"]

    # -----------------------------------------------------------
    def aerothermodynamics(self,
                           T_inf: float,
                           p_inf: float,
                           T_allow: float,
                           emissivity: float,
                           Pr: float = 0.72,
                           safety_factor: float = 1.2,
                           resample: int = 200,
                           n_theta: int = 4000) -> None:
        """Run the inviscid + viscous + blunt-LE pipeline and store the
        results on the instance.

        Parameters
        ----------
        T_inf   : freestream static temperature (K).
        p_inf   : freestream static pressure   (Pa).
        T_allow : maximum allowable wall temperature (K) — used for the
                  stagnation-heating bluntness sizing and as the constant
                  wall T for the Walz boundary-layer integration.
        emissivity : wall emissivity — used for the equilibrium wall
                     temperature calculation.
        Pr      : Prandtl number — used for the Walz boundary-layer integration.
        safety_factor : safety factor for the bluntness sizing.
        resample    : per-streamline resampling resolution.
        n_theta     : number of polar angle samples for Taylor--Maccoll profiles.
        """

        # Get freestream and shock conditions
        self.T_inf   = T_inf
        self.p_inf   = p_inf
        self.T_allow = T_allow
        self.rho_inf = p_inf / (R_AIR * T_inf)
        self.V_inf   = self.M1 * (self.gamma * R_AIR * T_inf) ** 0.5
        lower_mesh = self.panel.lower_mesh
        upper_mesh = self.panel.upper_mesh

        # Compute inviscid aerodynamic contributions
        self.inviscid_aerodynamics()

        # Compute viscous skin-friction contributions
        self.viscous_forces = compute_skin_friction(
            self.geometry, lower_mesh, upper_mesh,
            T_inf=T_inf, p_inf=p_inf, T_w=T_allow, Pr=Pr,
            resample=resample, n_theta=n_theta,
        )

        # Compute blunt leading edge radius
        self.le_size      = minimum_blunting_radius(self.rho_inf,self.V_inf,T_allow=T_allow,safety_factor=safety_factor, emissivity=emissivity)
        self.R_n          = self.le_size["R_min"]
        self.T_stag_check = equilibrium_wall_temperature(self.rho_inf, self.V_inf, self.R_n, emissivity=emissivity)

        # Compute blunt leading edge force contributions
        S_ref = self.inviscid_forces["planform_area"]
        self.blunt_forces = blunt_leading_edge_force(
            self.geometry, R_n=self.R_n,
            rho_inf=self.rho_inf, V_inf=self.V_inf,
            M1=self.M1, gamma=self.gamma, S_ref=S_ref,
        )

        # Compute total aerodynamic coefficients
        self.CL_total = self.inviscid_forces["CL"] + self.blunt_forces["dCL"]
        self.CD_total = (self.inviscid_forces["CD"]
                         + self.viscous_forces["CDf"]
                         + self.blunt_forces["dCD"])
        self.LD_total = (self.CL_total / self.CD_total
                         if abs(self.CD_total) > 1e-12 else float("inf"))

    # -----------------------------------------------------------
    def report(self) -> None:
        """Print a human-readable summary of geometry + analysis results."""
        sc = self.geometry["shock_conditions"]
        print(f"\nShock conditions")
        print(f"  M2            = {sc['M2']:.4f}")
        print(f"  theta         = {sc['theta_deg']:.4f} deg")
        print(f"  cone angle    = {sc['cone_half_angle_deg']:.4f} deg")
        print(f"  cone length   = {self.L:.3f} m")

        print(f"\nTrailing edge:")
        print(f"  R1_frac       = {self.R1_frac:.3f}")
        print(f"  W2_frac       = {self.W2_frac:.3f}")
        print(f"  n_shape       = {self.n_shape:.3f}")
        print(f"  beta          = {self.beta:.3f} deg")

        print(f"\nGeometry Statistics")
        print(f"  Length          = {self.vehicle_length:.3f} m")
        print(f"  Height          = {self.panel.height:.3f} m")
        print(f"  Wetted area     = {self.panel.wetted_area:.3f} m^2")
        print(f"  Volume (approx) = {self.panel.volume:.3f} m^3")
        print(f"  Triangles       = {self.panel.n_triangles}  "
              f"({self.panel.n_lower} lower, {self.panel.n_upper} upper)")

        if self.inviscid_forces is None:
            print("\n(aerothermodynamics(...) has not been run yet)")
            return

        Cp = self.inviscid_forces["Cp"]
        tau_max = getMaxWallShearStress(self.viscous_forces)
        print(f"\nAerodynamic coefficients")
        print(f"  Cp mean       = {Cp.mean():.4f}  "
              f"range [{Cp.min():.4f}, {Cp.max():.4f}]")
        print(f"  CL            = {self.inviscid_forces['CL']:.4f}")
        print(f"  CD (inviscid) = {self.inviscid_forces['CD']:.4f}")
        print(f"  CDf lower     = {self.viscous_forces['CDf_lower']:.4f}  "
              f"(D_f = {self.viscous_forces['D_lower']:.1f} N)")
        print(f"  CDf upper     = {self.viscous_forces['CDf_upper']:.4f}  "
              f"(D_f = {self.viscous_forces['D_upper']:.1f} N)")
        print(f"  tau_max       = {tau_max:.4f} Pa")

        print(f"\nBlunt-leading-edge correction")
        print(f"  T_allow       = {self.T_allow:.1f} K")
        print(f"  R_n (min)     = {self.R_n*1e3:.2f} mm  "
              f"(q_allow = {self.le_size['q_allow']/1e6:.3f} MW/m^2)")
        print(f"  dCL (blunt)   = {self.blunt_forces['dCL']:.4f}")
        print(f"  dCD (blunt)   = {self.blunt_forces['dCD']:.4f}")

        print(f"\nTotals")
        LDi = self.inviscid_forces["L_over_D"]
        LDv = self.inviscid_forces["CL"] / (
            self.inviscid_forces["CD"] + self.viscous_forces["CDf"])
        print(f"  CL total      = {self.CL_total:.4f}")
        print(f"  CD total      = {self.CD_total:.4f}")
        print(f"  L/D inviscid  = {LDi:.4f}")
        print(f"  L/D w/ visc.  = {LDv:.4f}")
        print(f"  L/D total     = {self.LD_total:.4f}")

    # -----------------------------------------------------------
    def plot(self, output_dir: str) -> None:
        if self.inviscid_forces is None:
            raise RuntimeError(
                "Call aerothermodynamics(...) before plot().")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Writing plots...")

        sc = self.geometry["shock_conditions"]
        beta_rad  = sc["beta_rad"]
        Vr_i      = sc["Vr_i"]
        V_theta_i = sc["V_theta_i"]
        lower_mesh = self.panel.lower_mesh
        upper_mesh = self.panel.upper_mesh

        # Base-plane diagnostic plot
        self.teg.plot_baseplane(
            self._z_func, self._Rs, self.L, self.N,
            beta_rad, Vr_i, V_theta_i,
            save_path=os.path.join(output_dir, "baseplane.png"),
        )

        # 3-D geometry / streamline plot
        X_b = np.full_like(self._Y_p, self.L)
        self.tracer.plot_geometry(
            self.geometry,
            self._X_p, self._Y_p, self._Z_p,
            X_b, self._Y_p, self._Z_p,
            save_path=os.path.join(output_dir, "streamlines.png"),
        )

        # Geometry view grids (shaded + wireframe)
        plot_geometry_views_pv(
            lower_mesh, upper_mesh, style="wireframe",
            save_path=os.path.join(output_dir, "geometry_views.png"),
        )

        # Pressure coefficient
        plot_scalar_field_pv(
            lower_mesh=lower_mesh,
            lower_field=self.pressure["Cp"],
            upper_mesh=upper_mesh,
            upper_field=np.zeros(upper_mesh["triangles"].shape[0]),
            cmap="viridis",
            colorbar_label="Cp",
            upper_alpha=0.3,
            save_path=os.path.join(output_dir, "pressure_cp.png"),
        )

        # Skin friction coefficient
        cf_lo = skin_friction_on_mesh(self.viscous_forces, lower_mesh,
                                       self.geometry, "lower", "cf")
        cf_up = skin_friction_on_mesh(self.viscous_forces, upper_mesh,
                                       self.geometry, "upper", "cf")
        cf_all = np.concatenate([cf_lo, cf_up])
        plot_scalar_field_pv(
            lower_mesh=lower_mesh, lower_field=np.log10(cf_lo),
            upper_mesh=upper_mesh, upper_field=np.log10(cf_up),
            cmap="magma",
            colorbar_label="log(c_f)",
            vmin=float(np.log10(cf_all.min())),
            vmax=float(np.percentile(np.log10(cf_all), 98)),
            save_path=os.path.join(output_dir, "skin_friction_cf.png"),
        )

        # Momentum boundary-layer thickness
        d2_lo = skin_friction_on_mesh(self.viscous_forces, lower_mesh,
                                       self.geometry, "lower", "delta2")
        d2_up = skin_friction_on_mesh(self.viscous_forces, upper_mesh,
                                       self.geometry, "upper", "delta2")
        plot_scalar_field_pv(
            lower_mesh=lower_mesh, lower_field=d2_lo * 1e3,
            upper_mesh=upper_mesh, upper_field=d2_up * 1e3,
            cmap="viridis",
            colorbar_label="delta_2 [mm]",
            save_path=os.path.join(output_dir, "momentum_thickness.png"),
        )

        # Flowfield Mach contours on cutting planes
        plot_flowfield_slices_pv(
            self.geometry, lower_mesh, upper_mesh,
            field="mach", cmap="Blues", vehicle_length=self.vehicle_length,
            save_path=os.path.join(output_dir, "flowfield_mach.png"),
        )

        # Flowfield Temperature contours on cutting planes
        plot_flowfield_slices_pv(
            self.geometry, lower_mesh, upper_mesh,
            field="temperature", cmap="Reds", vehicle_length=self.vehicle_length,
            save_path=os.path.join(output_dir, "flowfield_temperature.png"),
        )
        print(f"All plots saved to {output_dir}")

    # -----------------------------------------------------------
    def interactive(self) -> None:
        """Launch an interactive matplotlib viewer of the geometry"""
        plot_scalar_field(
            self.panel.lower_mesh, np.ones(self.panel.lower_mesh["triangles"].shape[0]),
            self.panel.upper_mesh, np.zeros(self.panel.upper_mesh["triangles"].shape[0]),
            cmap="jet",
            colorbar_label="Mesh",
            interactive=True,
            vmax=1.0,vmin=0.0,
        )
