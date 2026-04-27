import numpy as np
from scipy.integrate import solve_ivp


class Taylor_Maccoll:
    def __init__(self, gamma) -> None:
        self.gamma = gamma

    def TM_eqn(self, theta, S):
        gamma = self.gamma
        A, A_pr = S

        tep1 = (A * A_pr**2 - (gamma - 1) / 2 * (1 - A**2 - A_pr**2) * (2 * A + A_pr / np.tan(theta)))
        tep2 = ((gamma - 1) / 2 * (1 - A**2 - A_pr**2) - A_pr**2)
        A_2nd = tep1 / tep2

        return [A_pr, A_2nd]

    def solver(self, beta_rad, Vr_i, V_theta_i):

        def event_cr(theta, S):
            return S[1]
        event_cr.terminal = True

        sol = solve_ivp(
            self.TM_eqn, [beta_rad, 1e-08], y0=(Vr_i, V_theta_i),
            method='RK45', events=event_cr, rtol=1e-08, atol=1e-10
        )
        return sol

    def cone_half_angle(self, beta_rad, Vr_i, V_theta_i):
        sol = self.solver(beta_rad, Vr_i, V_theta_i)
        return sol.t[-1]

    def tracing_solver(self, Vr_i, V_theta_i, thetas, theta_range):

        def event_cr2(theta, S):
            return S[1]
        event_cr2.terminal = True

        sol2 = solve_ivp(
            self.TM_eqn, thetas, y0=(Vr_i, V_theta_i),
            method='RK45', events=event_cr2, t_eval=theta_range, rtol=1e-08, atol=1e-10
        )
        return sol2
