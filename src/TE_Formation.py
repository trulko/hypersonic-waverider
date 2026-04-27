import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from taylor_maccoll_sol import Taylor_Maccoll


class TEG:
    def __init__(self, gamma) -> None:
        self.tm_i = Taylor_Maccoll(gamma)

    def te_curve(self, z_func, Rs, L, N):
        """Compute trailing-edge and leading-edge projection arrays."""

        def equation(y):
            return np.sqrt(y**2 + z_func(y)**2) - Rs

        y_up = fsolve(equation, 2.0)[0]

        y_plot = np.linspace(-y_up, y_up, N)
        z_plot = z_func(y_plot)

        y_plot_break = np.linspace(0, y_up, N)
        z_plot_break = z_func(y_plot_break)
        x_plot_break = L * (1 - (Rs - np.sqrt(y_plot_break**2 + z_plot_break**2)) / Rs)

        return y_plot, z_plot, x_plot_break, y_plot_break, z_plot_break

    def baseplane_data(self, Rs, L, N, beta_rad, Vr_i, V_theta_i):
        """Return arrays needed to plot/inspect the base plane."""
        cone = self.tm_i.cone_half_angle(beta_rad, Vr_i, V_theta_i)

        y_conical = np.linspace(-Rs, Rs, N)
        z_conical = np.sqrt(Rs**2 - y_conical**2)
        z_conical_sym = -np.sqrt(Rs**2 - y_conical**2)

        R_c = L * np.tan(cone)
        Y_c = np.linspace(-R_c, R_c, N)
        Z_c = np.sqrt(R_c**2 - Y_c**2)
        Z_c_sym = -np.sqrt(R_c**2 - Y_c**2)

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
        ax.legend(loc='best')

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return cone
