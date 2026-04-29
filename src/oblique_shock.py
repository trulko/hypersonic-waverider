import numpy as np

class Oblique_Shock:
    def __init__(self) -> None:
        pass
    
    def sub_1(self, M1, gamma, beta):
        beta_rad = np.radians(beta)

        tan_theta = 2 * (1 / np.tan(beta_rad)) * ((M1**2 * np.sin(beta_rad)**2 - 1) / (M1**2 * (gamma + np.cos(2 * beta_rad)) + 2))
        theta_rad = np.atan(tan_theta)
        
        Mn1 = M1 * np.sin(beta_rad)
        
        tep1 = (2 / (gamma - 1))
        tep2 = (2 * gamma) / (gamma - 1)
        Mn2_sqr = (Mn1**2 + tep1) / (tep2 * Mn1**2 - 1)
        Mn2 = np.sqrt(Mn2_sqr)
        
        M2 = Mn2 / np.sin(beta_rad - theta_rad)
        delt = (theta_rad * 180)/np.pi

        return M2, delt, beta_rad, theta_rad

    def initial_nondimensioned_conditions(self, M1, gamma, beta):

        aftershock = self.sub_1(M1, gamma, beta)
        M = aftershock[0]
        V_pr = 1/(np.sqrt(2/((gamma - 1) * M**2) + 1))

        #   Initial Non-dimensioned Conditions
        Vr_i = V_pr * np.cos(aftershock[2]-aftershock[3])
        V_theta_i = - V_pr * np.sin(aftershock[2]-aftershock[3])

        return Vr_i, V_theta_i

    def test(self):
        print(self.M1, self.gamma, self.beta)
