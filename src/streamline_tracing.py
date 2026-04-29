import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

from taylor_maccoll_sol import Taylor_Maccoll
from TE_Formation import TEG


class TRACE:
    def __init__(self, gamma) -> None:
        self.tm_i = Taylor_Maccoll(gamma)
        self.TEG = TEG(gamma)

    def projection_module(self, z_func, Rs, L, N):
        y_plot, z_plot, x_plot_break, y_plot_break, z_plot_break = \
            self.TEG.te_curve(z_func, Rs, L, N)

        X_p = L * (1 - (Rs - np.sqrt(y_plot**2 + z_plot**2)) / Rs)
        Y_p = y_plot
        Z_p = z_plot

        X_b = np.full_like(y_plot, L)
        Y_b = y_plot
        Z_b = z_plot

        return X_b, Y_b, Z_b, X_p, Y_p, Z_p, x_plot_break, y_plot_break, z_plot_break

    def tracing_module(self, z_func, Rs, L, N, N_l, N_up, Vr_i, V_theta_i):
        X_b, Y_b, Z_b, X_p, Y_p, Z_p, x_plot_break, y_plot_break, z_plot_break = \
            self.projection_module(z_func, Rs, L, N)

        baseplane_x, baseplane_y, baseplane_z, baseplane_z_mir, baseplane_y_mir = [], [], [], [], []
        lower_surface = []

        r_i_s_full = np.sqrt(x_plot_break**2 + z_plot_break**2 + y_plot_break**2)
        theta_i_s_full = np.arctan(np.sqrt(z_plot_break**2 + y_plot_break**2) / x_plot_break)

        # Build evenly-spaced indices along the break curve
        idx = [0]
        for i in range(1, N_l + 1):
            j = len(r_i_s_full) % N_l
            if j == 0:
                idx.append(int(i * (len(r_i_s_full) / N_l) - 1))
            else:
                idx.append(int(i * ((len(r_i_s_full) - j) / N_l) - 1))

        for temp in range(N_l):
            phi = -np.arctan(y_plot_break[idx[temp]] / z_plot_break[idx[temp]])

            thetas = [np.abs(theta_i_s_full[idx[temp]]), 1e-08]
            theta_range = np.linspace(np.abs(theta_i_s_full[idx[temp]]), 1e-08, 1000)
            sol2 = self.tm_i.tracing_solver(Vr_i, V_theta_i, thetas, theta_range)

            S2 = [sol2.y[0][i] / sol2.y[1][i] for i in range(len(sol2.t))]
            Vr_Vtheta_ratio = np.array(S2)
            thet_array = np.array(sol2.t)

            r_march = [r_i_s_full[idx[temp]]]
            thet_march = [thet_array[0]]

            for i in range(1, len(thet_array)):
                d_theta = thet_array[i] - thet_array[i - 1]
                r_cur = r_march[-1]
                dr_dtheta = r_cur * Vr_Vtheta_ratio[i - 1]
                r_n = r_cur + dr_dtheta * d_theta

                if r_cur * np.cos(thet_array[i - 1]) - L < 0 <= r_n * np.cos(thet_array[i]) - L:
                    dx = r_n * np.cos(thet_array[i]) - r_cur * np.cos(thet_array[i - 1])
                    dz_rsin = r_n * np.sin(thet_array[i]) - r_cur * np.sin(thet_array[i - 1])
                    z_itp = np.cos(phi) * (dz_rsin / dx) * L + \
                            np.cos(phi) * (r_cur * np.sin(thet_array[i - 1]) -
                                           r_cur * np.cos(thet_array[i - 1]) * dz_rsin / dx)
                    r_break = np.sqrt((z_itp / np.cos(phi))**2 + L**2)
                    theta_break = np.arctan((z_itp / np.cos(phi)) / L)
                    r_march.append(r_break)
                    thet_march.append(theta_break)
                    break
                else:
                    r_march.append(r_n)
                    thet_march.append(thet_array[i])

            carte_x, carte_y, carte_z, carte_y_mir = [], [], [], []
            for i in range(len(r_march)):
                r_m = r_march[i]
                t_m = thet_march[i]
                carte_x.append(r_m * np.cos(t_m))
                carte_z.append(-r_m * np.sin(t_m) * np.cos(phi))
                carte_y.append(r_m * np.sin(t_m) * np.sin(phi))
                carte_y_mir.append(-r_m * np.sin(t_m) * np.sin(phi))

            baseplane_z.append(carte_z[-1])
            baseplane_z_mir.append(carte_z[-1])
            baseplane_y.append(carte_y[-1])
            baseplane_y_mir.append(-carte_y[-1])
            baseplane_x.append(carte_x[-1])

            carte_x[-1] = L
            crv = list(zip(carte_x, carte_y, carte_z))
            crv_mir = list(zip(carte_x, carte_y_mir, carte_z))
            lower_surface.append({"curve": crv, "mirrored_curve": crv_mir})

        # Assemble baseplane trailing-edge curve
        baseplane_z.append(z_plot_break[-1])
        baseplane_z_mir.append(z_plot_break[-1])
        baseplane_z.reverse()
        baseplane_z_mir.pop(0)
        m_disp_z = baseplane_z + baseplane_z_mir

        baseplane_y.append(y_plot_break[-1])
        baseplane_y_mir.append(-y_plot_break[-1])
        baseplane_y.reverse()
        baseplane_y_mir.pop(0)
        m_disp_y = baseplane_y + baseplane_y_mir

        m_disp_x = baseplane_x + baseplane_x
        m_disp_x.append(x_plot_break[-1])

        tck, u = splprep([m_disp_x, m_disp_y, m_disp_z], s=0)
        u_fine = np.linspace(0, 1, 1000)
        x_sm, y_sm, z_sm = splev(u_fine, tck)

        # Upper surface lines (free-stream streamlines)
        idx_up = [0]
        for i in range(1, N_up):
            j = len(r_i_s_full) % N_up
            if j == 0:
                idx_up.append(int(i * (len(r_i_s_full) / N_up) - 1))
            else:
                idx_up.append(int(i * ((len(r_i_s_full) - j) / N_up) - 1))

        upper_surface = []
        for temp2 in range(N_up):
            x_u = np.linspace(x_plot_break[idx_up[temp2]], L, N)
            y_u = y_plot_break[idx_up[temp2]]
            z_u = z_plot_break[idx_up[temp2]]
            upper_surface.append({
                "x": x_u.tolist(),
                "y": float(y_u),
                "z": float(z_u),
            })

        geometry = {
            "leading_edge": {"x": X_p.tolist(), "y": Y_p.tolist(), "z": Z_p.tolist()},
            "trailing_edge": {"x": X_b.tolist(), "y": Y_b.tolist(), "z": Z_b.tolist()},
            "lower_surface": lower_surface,
            "upper_surface": upper_surface,
            "baseplane_curve": {"x": x_sm.tolist(), "y": y_sm.tolist(), "z": z_sm.tolist()},
        }
        return geometry

    def plot_geometry(self, geometry, X_p, Y_p, Z_p, X_b, Y_b, Z_b, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for ls in geometry["lower_surface"]:
            cx, cy, cz = zip(*ls["curve"])
            cy_mir = [p[1] for p in ls["mirrored_curve"]]
            ax.plot(cx, cy, cz, color='b')
            ax.plot(cx, cy_mir, cz, color='b')

        bp = geometry["baseplane_curve"]
        ax.plot(bp["x"], bp["y"], bp["z"], color='r', lw=2, label='Lower Surface TE')

        ax.plot(X_p, Y_p, Z_p, color='k', label='Leading Edge')
        ax.plot(X_b, Y_b, Z_b, color='y', label='Trailing Edge', zorder=20)

        for us in geometry["upper_surface"]:
            x_u = us["x"]
            y_u, z_u = us["y"], us["z"]
            ax.plot(x_u, [y_u] * len(x_u), [z_u] * len(x_u), color='c')
            ax.plot(x_u, [-y_u] * len(x_u), [z_u] * len(x_u), color='c')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc='best')
        ax.set_aspect('equal')

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
