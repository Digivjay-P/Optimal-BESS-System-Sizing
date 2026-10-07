import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. PARSE UPLOADED DATA FILES
# ==========================================
# Parse Load Profile
load_df = pd.read_csv('load profile.csv', encoding='latin1')
load_profile = np.zeros(24)

for index, row in load_df.iterrows():
    time_str = str(row['Time'])
    load_str = str(row['Load (kW)'])
    
    t_match = re.findall(r'\d+', time_str)
    if len(t_match) == 2:
        t_start, t_end = int(t_match[0]), int(t_match[1])
        
        l_match = re.findall(r'\d+', load_str)
        if len(l_match) == 2:
            l_avg = (float(l_match[0]) + float(l_match[1])) / 2.0
        elif len(l_match) == 1:
            l_avg = float(l_match[0])
        else:
            l_avg = 0
            
        for h in range(t_start, t_end):
            if h < 24:
                load_profile[h] = l_avg

# Parse Irradiance
irr_df = pd.read_csv('irradiance.csv', encoding='latin1')
annual_globinc = float(irr_df.iloc[15, 4]) 
daily_psh = annual_globinc / 365.0

# Generate 24-hour PV profile (Sine wave approximation based on PSH)
pv_profile_1kw = np.zeros(24)
daylight_hours = np.arange(6, 18)
pv_shape = np.sin(np.pi * (daylight_hours - 6) / 12)
pv_profile_1kw[6:18] = pv_shape * (daily_psh / np.sum(pv_shape))

PV_CAPACITY_KW = 200.0
pv_profile = pv_profile_1kw * PV_CAPACITY_KW

# ==========================================
# 2. SYSTEM PARAMETERS
# ==========================================
peak_load = np.max(load_profile)
critical_load = 0.40 * peak_load 
COST_PER_KWH = 12000.0      
COST_PER_KW = 8000.0        
SOC_MAX = 0.95               
SOC_EFFECTIVE_MIN = 0.20
ROUND_TRIP_EFF = 0.90

# Calculate daily surplus energy available for charging
surplus_energy = 0
for t in range(24):
    if pv_profile[t] > load_profile[t]:
        surplus_energy += (pv_profile[t] - load_profile[t]) * np.sqrt(ROUND_TRIP_EFF)

# ==========================================
# 3. CONSTRAINED SIZING SWEEP
# ==========================================
autonomy_days_sweep = np.linspace(0.1, 2.0, 10)
results_capex = []
results_bess_cap = []
results_days_charge = []

# Calculate exact nighttime load (6 PM to 6 AM)
night_load = sum(load_profile) - sum(load_profile[6:18]) 

for days in autonomy_days_sweep:
    hours = days * 24.0
    
    # Constraint 1: The "Nighttime Survival" Floor
    # Battery must be large enough to survive normal night loads regardless of autonomy
    min_E_night = (night_load) / (SOC_MAX - SOC_EFFECTIVE_MIN) / np.sqrt(ROUND_TRIP_EFF)
    
    # Constraint 2: The "Critical Load Autonomy" Requirement
    min_required_E = (critical_load * hours) / (SOC_MAX - SOC_EFFECTIVE_MIN)
    
    # The optimizer is mathematically bound by whichever constraint is larger
    opt_E_cap = max(min_E_night, min_required_E)
    opt_P_cap = peak_load # Droop / Black start bound
    
    capex = (opt_E_cap * COST_PER_KWH) + (opt_P_cap * COST_PER_KW)
    
    # Calculate days required to charge from empty
    usable_capacity = opt_E_cap * (SOC_MAX - SOC_EFFECTIVE_MIN)
    days_of_charge = usable_capacity / surplus_energy if surplus_energy > 0 else float('inf')
    
    results_capex.append(capex / 1e5)
    results_bess_cap.append(opt_E_cap)
    results_days_charge.append(days_of_charge)

# ==========================================
# 4. PLOTTING
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(autonomy_days_sweep, results_capex, marker='o', color='b', linewidth=2)
ax1.set_title('System Cost vs. Days of Autonomy')
ax1.set_xlabel('Days of Autonomy')
ax1.set_ylabel('Total Initial CAPEX (₹ Lakhs)')
ax1.grid(True, linestyle='--')

ax2.plot(results_bess_cap, results_capex, marker='s', color='r', linewidth=2)
ax2.set_title('System Cost vs. Battery Capacity')
ax2.set_xlabel('Optimized BESS Capacity (kWh)')
ax2.set_ylabel('Total Initial CAPEX (₹ Lakhs)')
ax2.grid(True, linestyle='--')

ax3.plot(results_days_charge, results_capex, marker='^', color='g', linewidth=2)
ax3.set_title('System Cost vs. Days of Charge')
ax3.set_xlabel('Required Days to Fully Charge BESS')
ax3.set_ylabel('Total Initial CAPEX (₹ Lakhs)')
ax3.grid(True, linestyle='--')

plt.tight_layout()
plt.show()