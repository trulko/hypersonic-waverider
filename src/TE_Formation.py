import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from taylor_maccoll_sol import Taylor_Maccoll

class TEG:
    R1_FRAC_MIN = 0.20
    R1_FRAC_MAX = 0.85
    W_FRAC_MIN = 0.25
    W_FRAC_MAX = 0.85
    N_SHAPE_MIN = 0.60
    N_SHAPE_MAX = 10.0

    def __init__(self, gamma) -> None:
        self.tm_i = Taylor_Maccoll(gamma)

    def make_simple_backface(self, L, beta_deg, R1_frac=0.20, W_frac=0.80, n_shape=2.0):
        """
        Build a callable z_func(y) for a simple admissible trailing-edge shape.

        Parameters
        ----------
        L : float
            Vehicle length [m]
        beta_deg : float
            Shock semi-angle in degrees
        R1_frac : float
            Centerline radial fraction R1 / Rs
            Required bound: 0.20 <= R1_frac <= 0.85
        W_frac : float
            Half-span fraction y_up / Rs
            Required bound: 0.25 <= W_frac <= 0.85
        n_shape : float
            Shape exponent
            Required bound: 0.6 <= n_shape <= 10

        Returns
        -------
        z_func : callable
            Function handle z = z_func(y), valid for |y| <= y_up
        Rs : float
            Shock radius in the base plane
        """

        # -----------------------------
        # Basic input checks
        # -----------------------------
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError("L must be positive.")
        if not np.isfinite(beta_deg) or not (0.0 < beta_deg < 89.0):
            raise ValueError("beta_deg must lie in (0, 89).")
        if not np.isfinite(R1_frac):
            raise ValueError("R1_frac must be finite.")
        if not np.isfinite(W_frac):
            raise ValueError("W_frac must be finite.")
        if not np.isfinite(n_shape):
            raise ValueError("n_shape must be finite.")
        if not (self.R1_FRAC_MIN <= R1_frac <= self.R1_FRAC_MAX):
            raise ValueError(
                f"R1_frac must satisfy "
                f"{self.R1_FRAC_MIN:.2f} <= R1_frac <= {self.R1_FRAC_MAX:.2f}."
            )
        if not (self.W_FRAC_MIN <= W_frac <= self.W_FRAC_MAX):
            raise ValueError(
                f"W_frac must satisfy "
                f"{self.W_FRAC_MIN:.2f} <= W_frac <= {self.W_FRAC_MAX:.2f}."
            )
        if not (self.N_SHAPE_MIN <= n_shape <= self.N_SHAPE_MAX):
            raise ValueError(
                f"n_shape must satisfy "
                f"{self.N_SHAPE_MIN:.1f} <= n_shape <= {self.N_SHAPE_MAX:.1f}."
            )

        # Build geometry
        beta_rad = np.radians(beta_deg)
        Rs = float(L * np.tan(beta_rad))
        R1 = float(R1_frac * Rs)
        y_up = float(W_frac * Rs)
        gap0 = float(Rs**2 - R1**2)

        def z_func(y):
            """
            Callable back-face function.
            Valid for |y| <= y_up.
            """
            y_arr = np.asarray(y, dtype=float)
            scalar_input = (y_arr.ndim == 0)
            y_abs = np.abs(y_arr)
            if np.any(y_abs > y_up + 1e-12):
                raise ValueError(
                    f"y outside admissible domain [-{y_up:.6g}, {y_up:.6g}]"
                )
            eta = (y_abs / y_up) ** 2
            base = np.maximum(0.0, 1.0 - eta)
            gap = gap0 * base**n_shape
            radicand = Rs**2 - y_abs**2 - gap

            # Numerical guard
            if np.any(radicand < -1e-9):
                raise ValueError(
                    "Negative radicand encountered. "
                    "This indicates a non-admissible or numerically unstable shape."
                )
            radicand = np.maximum(radicand, 0.0)
            z_val = -np.sqrt(radicand)
            if scalar_input: return float(z_val)
            return z_val

        # Attach metadata directly to the function handle
        z_func.Rs = Rs
        z_func.y_up = y_up
        z_func.R1 = R1
        z_func.beta_deg = float(beta_deg)
        z_func.report = {
            "ok": True,
            "L": float(L),
            "beta_deg": float(beta_deg),
            "Rs": Rs,
            "R1": R1,
            "y_up": y_up,
            "R1_frac": float(R1_frac),
            "W_frac": float(W_frac),
            "n_shape": float(n_shape),
            "z_center": float(z_func(0.0)),
            "z_tip": float(z_func(y_up)),
            "tip_slope_singular": bool(n_shape < 1.0),
            "tip_tangent_to_shock": bool(n_shape > 1.0),
        }

        # Optional sampled sanity check
        y_chk = np.linspace(0.0, y_up, 2001)
        z_chk = z_func(y_chk)
        rho_chk = np.sqrt(y_chk**2 + z_chk**2)

        if np.any(rho_chk > Rs * (1.0 + 1e-9)):
            raise ValueError("Generated back-face leaves the shock circle.")

        if np.any(~np.isfinite(z_chk)):
            raise ValueError("Generated back-face contains non-finite values.")

        return z_func, Rs

    @staticmethod
    def _extract_shape_metadata(z_func):
        """
        Try to recover y_up / Rs from:
        - a plain function with attached attributes
        - a bound method whose owner stores attributes
        """
        owner = getattr(z_func, "__self__", None)

        y_up = getattr(z_func, "y_up", None)
        if y_up is None and owner is not None:
            y_up = getattr(owner, "y_up", None)

        Rs = getattr(z_func, "Rs", None)
        if Rs is None and owner is not None:
            Rs = getattr(owner, "Rs", None)

        return y_up, Rs

    def te_curve(self, z_func, Rs, L, N):
        """Compute trailing-edge and leading-edge projection arrays."""

        meta_y_up, meta_Rs = self._extract_shape_metadata(z_func)

        if meta_Rs is not None:
            Rs = float(meta_Rs)

        # Prefer known y_up from the generated function handle
        if meta_y_up is not None:
            y_up = float(meta_y_up)
        else:
            # Fallback for legacy z_func input
            def equation(y):
                z = z_func(y)
                z = float(np.asarray(z).reshape(-1)[0])
                return np.sqrt(y**2 + z**2) - Rs
            g0 = equation(0.0)
            g1 = equation(Rs)
            if g0 > 0.0:
                raise ValueError("Trailing-edge centerline lies outside the shock circle.")
            if g1 < 0.0:
                raise ValueError(
                    "Could not bracket TE/shock intersection on [0, Rs]. "
                    "Check the supplied z_func."
                )
            y_up = float(brentq(equation, 0.0, Rs))

        y_plot = np.linspace(-y_up, y_up, N)
        z_plot = z_func(y_plot)

        y_plot_break = np.linspace(0.0, y_up, N)
        z_plot_break = z_func(y_plot_break)

        rho_break = np.sqrt(y_plot_break**2 + z_plot_break**2)
        x_plot_break = L * rho_break / Rs

        return y_plot, z_plot, x_plot_break, y_plot_break, z_plot_break

    def baseplane_data(self, Rs, L, N, beta_rad, Vr_i, V_theta_i):
        """Return arrays needed to plot/inspect the base plane."""
        cone = self.tm_i.cone_half_angle(beta_rad, Vr_i, V_theta_i)

        y_conical = np.linspace(-Rs, Rs, N)
        z_conical = np.sqrt(np.maximum(0.0, Rs**2 - y_conical**2))
        z_conical_sym = -z_conical

        R_c = L * np.tan(cone)
        Y_c = np.linspace(-R_c, R_c, N)
        Z_c = np.sqrt(np.maximum(0.0, R_c**2 - Y_c**2))
        Z_c_sym = -Z_c

        return cone, y_conical, z_conical, z_conical_sym, Y_c, Z_c, Z_c_sym

    def plot_baseplane(self, z_func, Rs, L, N, beta_rad, Vr_i, V_theta_i, save_path=None):
        y_plot, z_plot, *_ = self.te_curve(z_func, Rs, L, N)
        cone, y_conical, z_conical, z_conical_sym, Y_c, Z_c, Z_c_sym = \
            self.baseplane_data(Rs, L, N, beta_rad, Vr_i, V_theta_i)

        fig, ax = plt.subplots()
        ax.plot(y_conical, z_conical, color='r', label='Conical Shock')
        ax.plot(y_conical, z_conical_sym, color='r')
        ax.plot(Y_c, Z_c, color='k', label='Base Cone')
        ax.plot(Y_c, Z_c_sym, color='k')
        ax.plot(y_plot, z_plot, label='Trailing Edge')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.set_title('Base Plane')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend(loc='best',frameon=False)

        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        return cone
