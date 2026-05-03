import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Constants and Given Parameters
# ==========================================
k_SG = 1.83e-4          # Sutton-Graves constant [W s^3 kg^-1/2 m^-7/2]
sigma = 5.67e-8         # Stefan-Boltzmann constant [W / (m^2 K^4)]
T_bg = 300.0            # Background temperature [K]
s_f = 1.5               # Safety factor
eps = 0.9               # Emissivity
gamma = 1.4             # Specific heat ratio for air
R = 287.05              # Specific gas constant for air [J/(kg K)]

# ==========================================
# US Standard Atmosphere 1976 (up to 25 km)
# ==========================================
def std_atmosphere(altitude_ft):
    """
    Computes density and speed of sound based on standard atmosphere.
    """
    altitude_m = altitude_ft * 0.3048  # Convert feet to meters
    
    # 0 to 11 km (Troposphere)
    if altitude_m < 11000:
        T = 288.15 - 0.0065 * altitude_m
        P = 101325 * (T / 288.15) ** 5.25588
    # 11 km to 20 km (Lower Stratosphere / Tropopause)
    elif altitude_m < 20000:
        T = 216.65
        P = 22632.1 * np.exp(-9.80665 * (altitude_m - 11000) / (R * 216.65))
    # 20 km to 32 km (Upper Stratosphere)
    else:
        T = 216.65 + 0.001 * (altitude_m - 20000)
        P = 5474.89 * (216.65 / T) ** 34.16319
        
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    return rho, a

# ==========================================
# Blunting Radius Function
# ==========================================
def compute_Rn_min(T_allow, rho_inf, V_inf):
    """
    Computes the minimum blunting radius based on aerothermal heating constraints.
    """
    numerator = s_f * k_SG * V_inf**3
    denominator = eps * sigma * (T_allow**4 - T_bg**4)
    return rho_inf * (numerator / denominator)**2

# ==========================================
# Flight Conditions
# ==========================================
mach_number = 6.0

# Condition 1: 50,000 ft
rho_50k, a_50k = std_atmosphere(50000)
V_50k = mach_number * a_50k

# Condition 2: 70,000 ft
rho_70k, a_70k = std_atmosphere(70000)
V_70k = mach_number * a_70k

# Generate allowable temperature range (e.g., 1000 K to 3000 K)
T_allow = np.linspace(1000, 3000, 500)

# Calculate Rn,min
Rn_50k = compute_Rn_min(T_allow, rho_50k, V_50k)
Rn_70k = compute_Rn_min(T_allow, rho_70k, V_70k)

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(9, 6))

# Plot both curves
plt.plot(T_allow, Rn_50k, label=f'Mach 6 at 50,000 ft\n($\\rho_\\infty={rho_50k:.4f}$ kg/m$^3$, $V_\\infty={V_50k:.1f}$ m/s)', color='firebrick', linewidth=2)
plt.plot(T_allow, Rn_70k, label=f'Mach 6 at 70,000 ft\n($\\rho_\\infty={rho_70k:.4f}$ kg/m$^3$, $V_\\infty={V_70k:.1f}$ m/s)', color='navy', linewidth=2)

# Graph Formatting
plt.yscale('log') # Log scale helps capture the power of 8 dropoff 
plt.xlabel('Allowable Surface Temperature, $T_{\\text{allow}}$ (K)', fontsize=12)
plt.ylabel('Minimum Blunting Radius, $R_{n,\\min}$ (m)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11,frameon=False)
plt.xlim(min(T_allow), max(T_allow))

# Display plot
plt.tight_layout()
plt.show()