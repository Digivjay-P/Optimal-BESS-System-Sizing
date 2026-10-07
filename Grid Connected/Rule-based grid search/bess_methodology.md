# Basic PV+BESS Optimizer Methodology

## Inputs Used

- PV profile: Monthly PVsyst `E_Grid` energy from `irradiance.csv`
- Load profile: Daily time-block load ranges from `load profile.csv`
- Import tariff: Rs 8/kWh
- Export credit: Rs 0/kWh, assumed conservative for the initial trial
- PV rating: 1 MW
- Battery fixed case: 1 MW / 2 MWh
- Initial SOC: 50%
- SOC range: 20% to 90%
- Battery efficiency: 92%
- PV cost: Rs 40 million/MW
- Battery cost: Rs 15,000/kWh
- Assumed inverter cost: Rs 4 million/MW
- Assumed land and installation cost: Rs 5 million/MW
- Financial life: 10 years
- Discount rate: 8%

## Why This Method

The first trial uses a rule-based hourly simulation with a grid search over battery capacity. This is the right starting point because it is transparent and easy to debug. A more advanced optimizer such as MILP can be added later after the input structure and economic assumptions are confirmed.

## Data Preparation

The PVsyst file is monthly, not hourly. To run an hourly battery dispatch, each month's `E_Grid` total is distributed across daylight hours using a simple sinusoidal production shape. The load file gives time blocks and ranges, so each block is expanded to hourly values using the midpoint of the range.

## Dispatch Rule

For every hour:

1. PV generation serves load first.
2. Surplus PV charges the battery within SOC and power limits.
3. Remaining PV is exported.
4. If load remains, the battery discharges within SOC and power limits.
5. Any final deficit is imported from the grid.

Grid charging is not used in this first trial.

## Main Result

The synthetic annual profile gives:

- Annual PV available: 612,154 kWh
- Annual load: 262,982.5 kWh
- No-battery grid import: 169,312.4 kWh
- No-battery annual grid bill: Rs 1,354,499.5

The lowest annualized cost among tested capacities is the no-battery case. The fixed 1 MW / 2 MWh BESS removes grid import, but the avoided annual bill is only about Rs 1.35 million while the battery plus inverter annualized cost is about Rs 5.07 million. On this initial profile, the simple payback is about 25.1 years.

## Recharge-Days Cost Charts

Two cost charts are produced with the capacity sweep. "Equivalent recharge days" means the PV-surplus days needed to charge a BESS from 20% to 90% SOC, calculated as:

`battery capacity x (90% - 20%) / sqrt(92% efficiency) / average daily no-BESS PV export`

The current no-BESS PV surplus is about 1,420.5 kWh/day, so the fixed 2 MWh BESS requires about 1.03 equivalent recharge days. This is an average-energy measure, not a guarantee that every month has enough surplus energy.

The cost assumptions in the charts are Rs 8/kWh import tariff, Rs 0/kWh export credit, Rs 15,000/kWh battery cost, Rs 4 million/MW inverter cost, Rs 5 million/MW land and installation cost, 10-year financial life, and 8% discount rate. PV cost is Rs 40 million/MW. The optimum is printed by the script and marked on the total-cost chart.

## Days of Autonomy Cost Charts

In addition to recharge days, two cost charts are generated with respect to "Days of Autonomy":
- `cost_vs_days_of_autonomy.svg`: Annualized Total Project Cost vs Days of Autonomy
- `cost_components_vs_days_of_autonomy.svg`: Annual Grid Bill, Annualized BESS Cost, and Total Cost vs Days of Autonomy

"Days of Autonomy" represents how long (in days) the battery energy storage system can sustain the daily load demand independently from 90% SOC down to 20% SOC without any solar or grid support, calculated as:

`battery capacity x (90% - 20%) x sqrt(92% efficiency) / average daily load`

With the average daily load of 720.5 kWh/day:
- 0 kWh: 0.00 days (0.0 hours)
- 500 kWh: 0.47 days (11.2 hours)
- 1,000 kWh: 0.93 days (22.4 hours)
- 1,500 kWh: 1.40 days (33.5 hours)
- 2,000 kWh (Fixed 2 MWh case): 1.86 days (44.7 hours)
- 2,500 kWh: 2.33 days (55.9 hours)
- 3,000 kWh: 2.80 days (67.1 hours)
- 4,000 kWh: 3.73 days (89.5 hours)

The lowest annualized cost overall remains 0 kWh (grid reliance, Rs 8.06M/yr). Among configurations with storage, 500 kWh provides 0.47 days (11.2 hours) of autonomy at Rs 8.35M/yr with an 8.66-year payback. A 1,000 kWh BESS achieves 0.93 days (~1 day) of autonomy and eliminates all grid imports under typical dispatch.

## Important Limitation

The PV input is monthly, so the hourly PV shape is synthetic. For a better optimizer, use hourly PVsyst output if available, especially `EArray` or `E_Grid` at hourly resolution.
