"""
Continuous Battery Sizing and MILP/LP Optimal Dispatch Framework
-----------------------------------------------------------------
Features:
- Continuous battery capacity evaluation across 1 to 1,000 kWh.
- Dynamic Time-of-Day (ToD) Grid Import Tariff (Madhya Pradesh MPERC benchmark).
- Variable Time-Dependent Grid Export Tariff (Solar Net Billing / APPC benchmark).
- True mathematical optimal dispatch via SciPy HiGHS LP solver over all 8,760 hours.
- Computes Net Grid Bill, Export Revenue, Days of Autonomy, Recharge Days, Payback, and Total Project Cost.
- Generates high-resolution dependency-free SVG charts and detailed CSV/JSON reports.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
import numpy as np
import scipy.optimize as opt
from scipy.sparse import lil_matrix, csc_matrix

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

#INPUTS FOR THE SYSTEM ARE HERE AND CAN BE MODIFIED HERE.
@dataclass(frozen=True)
class ProjectAssumptions:
    pv_rating_mw: float = 0.25
    pv_cost_rs_per_mw: float = 40_000_000.0
    land_and_installation_rs_per_mw: float = 5_000_000.0
    battery_cost_rs_per_kwh: float = 15_000.0
    inverter_cost_rs_per_mw: float = 4_000_000.0
    battery_efficiency: float = 0.92
    min_soc_fraction: float = 0.20
    max_soc_fraction: float = 0.90
    initial_soc_fraction: float = 0.50
    project_years: int = 10
    discount_rate: float = 0.08
    storage_duration_hours: float = 2.0

#CALCULATES THE CAPITAL RECOVERY FACTOR FOR ANNUALIZED COSTS BASED ON DISCOUNT RATE AND PROJECT YEARS.
def capital_recovery_factor(rate: float, years: int) -> float:
    if rate == 0:
        return 1.0 / years
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def get_dynamic_tariffs(hour: int) -> tuple[float, float]:
    """
    Returns (import_tariff_rs_per_kwh, export_tariff_rs_per_kwh) for Madhya Pradesh.
    
    MPERC HT Consumer Time-of-Day (ToD) Import Tariff:
    - 18:00 - 22:00 (Peak): Rs 10.40 / kWh (+30% peak surcharge)
    - 22:00 - 06:00 (Off-Peak): Rs 6.50 / kWh (-18.75% night rebate)
    - 06:00 - 18:00 (Normal): Rs 8.00 / kWh (base rate)
    
    MPERC Solar Net Billing / Variable Export Tariff:
    - 10:00 - 16:00 (Midday solar peak): Rs 2.50 / kWh (APPC benchmark)
    - 06:00 - 10:00 (Morning shoulder): Rs 3.20 / kWh
    - 16:00 - 18:00 (Afternoon shoulder): Rs 3.80 / kWh
    - 18:00 - 22:00 (Evening peak export): Rs 5.50 / kWh
    - 22:00 - 06:00 (Night base): Rs 2.50 / kWh
    """
    # Import Tariff
    if 18 <= hour < 22:
        c_imp = 10.40
    elif 22 <= hour or hour < 6:
        c_imp = 6.50
    else:
        c_imp = 8.00
        
    # Export Tariff
    if 10 <= hour < 16:
        c_exp = 2.50
    elif 6 <= hour < 10:
        c_exp = 3.20
    elif 16 <= hour < 18:
        c_exp = 3.80
    elif 18 <= hour < 22:
        c_exp = 5.50
    else:
        c_exp = 2.50
        
    return c_imp, c_exp

#READ THE CSV FILES
def read_pvsyst_monthly(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = list(csv.reader(file))
    header = None
    data = {}
    for r in rows:
        if len(r) > 2 and r[1] == "GlobHor":
            header = ["Month", *r[1:]]
        elif header and r and r[0] in MONTHS:
            data[r[0]] = float(r[7])  # E_Grid column
    return data


def read_load_profile(path: Path) -> list[float]:
    hourly_load = [0.0] * 24
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            t_nums = [int(x) for x in re.findall(r"\d+", row["Time"])]
            l_nums = [float(x) for x in re.findall(r"[\d\.]+", row["Load (kW)"])]
            mid_load = (l_nums[0] + l_nums[1]) / 2.0
            for h in range(t_nums[0], t_nums[1]):
                hourly_load[h % 24] = mid_load
    return hourly_load


def build_annual_timeseries(
    pv_monthly: dict[str, float],
    daily_load: list[float],
    year: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pv_list = []
    load_list = []
    imp_list = []
    exp_list = []
    
    for m_num, m_name in enumerate(MONTHS, start=1):
        days = calendar.monthrange(year, m_num)[1]
        weights = []
        for _ in range(days):
            for h in range(24):
                weights.append(max(0.0, math.sin(math.pi * (h - 6) / 12.0)))
        scale = pv_monthly[m_name] / sum(weights) if sum(weights) else 0.0
        
        for d in range(days):
            for h in range(24):
                w = weights[d * 24 + h]
                pv_list.append(w * scale)
                load_list.append(daily_load[h])
                c_imp, c_exp = get_dynamic_tariffs(h)
                imp_list.append(c_imp)
                exp_list.append(c_exp)
                
    return np.array(pv_list), np.array(load_list), np.array(imp_list), np.array(exp_list)


@dataclass
class SizingResult:
    battery_capacity_kwh: float
    battery_power_kw: float
    grid_import_kwh: float
    grid_export_kwh: float
    grid_import_cost_rs: float
    grid_export_revenue_rs: float
    net_energy_bill_rs: float
    pv_to_load_kwh: float
    pv_to_battery_kwh: float
    battery_to_load_kwh: float
    curtailed_solar_kwh: float
    usable_discharge_energy_kwh: float
    days_of_autonomy: float
    hours_of_autonomy: float
    equivalent_recharge_days: float
    annualized_battery_cost_rs: float
    annualized_fixed_pv_cost_rs: float
    annualized_total_project_cost_rs: float
    annual_savings_vs_no_battery_rs: float
    simple_payback_years: float | None


def solve_dispatch_lp(
    pv: np.ndarray,
    load: np.ndarray,
    c_imp: np.ndarray,
    c_exp: np.ndarray,
    capacity_kwh: float,
    power_kw: float,
    assumptions: ProjectAssumptions,
) -> tuple[SizingResult, dict[str, np.ndarray]]:
    T = len(pv)
    eta_ch = math.sqrt(assumptions.battery_efficiency)
    eta_dis = math.sqrt(assumptions.battery_efficiency)
    crf = capital_recovery_factor(assumptions.discount_rate, assumptions.project_years)
    annualized_fixed_pv = (
        assumptions.pv_rating_mw * (assumptions.pv_cost_rs_per_mw + assumptions.land_and_installation_rs_per_mw) * crf
    )
    
    if capacity_kwh <= 0.0 or power_kw <= 0.0:
        # No battery case: direct analytical solution
        pv_to_load = np.minimum(pv, load)
        surplus = pv - pv_to_load
        deficit = load - pv_to_load
        
        grid_import_kwh = float(deficit.sum())
        grid_export_kwh = float(surplus.sum())
        grid_import_cost = float((deficit * c_imp).sum())
        grid_export_revenue = float((surplus * c_exp).sum())
        net_bill = grid_import_cost - grid_export_revenue
        
        summary = SizingResult(
            battery_capacity_kwh=0.0,
            battery_power_kw=0.0,
            grid_import_kwh=grid_import_kwh,
            grid_export_kwh=grid_export_kwh,
            grid_import_cost_rs=grid_import_cost,
            grid_export_revenue_rs=grid_export_revenue,
            net_energy_bill_rs=net_bill,
            pv_to_load_kwh=float(pv_to_load.sum()),
            pv_to_battery_kwh=0.0,
            battery_to_load_kwh=0.0,
            curtailed_solar_kwh=0.0,
            usable_discharge_energy_kwh=0.0,
            days_of_autonomy=0.0,
            hours_of_autonomy=0.0,
            equivalent_recharge_days=0.0,
            annualized_battery_cost_rs=0.0,
            annualized_fixed_pv_cost_rs=annualized_fixed_pv,
            annualized_total_project_cost_rs=net_bill + annualized_fixed_pv,
            annual_savings_vs_no_battery_rs=0.0,
            simple_payback_years=None,
        )
        dispatch_dict = {
            "pv_to_load": pv_to_load,
            "pv_to_battery": np.zeros(T),
            "pv_to_grid": surplus,
            "battery_to_load": np.zeros(T),
            "grid_to_load": deficit,
            "soc": np.zeros(T),
        }
        return summary, dispatch_dict

    # Variables:
    # 0..T-1:     p_pv_load
    # T..2T-1:    p_pv_bat
    # 2T..3T-1:   p_pv_grid
    # 3T..4T-1:   p_bat_load
    # 4T..5T-1:   p_grid_load
    # 5T..6T-1:   soc
    n_vars = 6 * T
    c = np.zeros(n_vars)
    c[4 * T : 5 * T] = c_imp   # minimize import cost
    c[2 * T : 3 * T] = -c_exp  # maximize export revenue
    
    # Equality Constraints:
    # 1) Load balance: p_pv_load[t] + p_bat_load[t] + p_grid_load[t] = load[t]
    # 2) SOC continuity: soc[t+1] - soc[t] - eta_ch*p_pv_bat[t] + p_bat_load[t]/eta_dis = 0
    A_eq = lil_matrix((2 * T, n_vars))
    b_eq = np.zeros(2 * T)
    for t in range(T):
        A_eq[t, t] = 1.0
        A_eq[t, 3 * T + t] = 1.0
        A_eq[t, 4 * T + t] = 1.0
        b_eq[t] = load[t]
        
        nxt = (t + 1) % T
        A_eq[T + t, 5 * T + nxt] = 1.0
        A_eq[T + t, 5 * T + t] = -1.0
        A_eq[T + t, T + t] = -eta_ch
        A_eq[T + t, 3 * T + t] = 1.0 / eta_dis
        b_eq[T + t] = 0.0

    # Inequality Constraints:
    # 1) PV balance: p_pv_load[t] + p_pv_bat[t] + p_pv_grid[t] <= pv[t]
    # 2) Battery charging power limit: p_pv_bat[t] <= power_kw
    # 3) Battery discharging power limit: p_bat_load[t] <= power_kw
    A_ub = lil_matrix((3 * T, n_vars))
    b_ub = np.zeros(3 * T)
    for t in range(T):
        A_ub[t, t] = 1.0
        A_ub[t, T + t] = 1.0
        A_ub[t, 2 * T + t] = 1.0
        b_ub[t] = pv[t]
        
        A_ub[T + t, T + t] = 1.0
        b_ub[T + t] = power_kw
        
        A_ub[2 * T + t, 3 * T + t] = 1.0
        b_ub[2 * T + t] = power_kw
        
    bounds = [(0.0, None)] * (5 * T) + [(assumptions.min_soc_fraction * capacity_kwh, assumptions.max_soc_fraction * capacity_kwh)] * T
    
    res = opt.linprog(
        c,
        A_ub=A_ub.tocsc(),
        b_ub=b_ub,
        A_eq=A_eq.tocsc(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP optimization failed for capacity {capacity_kwh}: {res.message}")
        
    sol = res.x
    p_pv_load = sol[:T]
    p_pv_bat = sol[T : 2 * T]
    p_pv_grid = sol[2 * T : 3 * T]
    p_bat_load = sol[3 * T : 4 * T]
    p_grid_load = sol[4 * T : 5 * T]
    soc = sol[5 * T :]
    
    grid_import_kwh = float(p_grid_load.sum())
    grid_export_kwh = float(p_pv_grid.sum())
    grid_import_cost = float((p_grid_load * c_imp).sum())
    grid_export_revenue = float((p_pv_grid * c_exp).sum())
    net_bill = grid_import_cost - grid_export_revenue
    
    battery_capex = capacity_kwh * assumptions.battery_cost_rs_per_kwh + power_kw * (assumptions.inverter_cost_rs_per_mw / 1000.0)
    annualized_battery_cost = battery_capex * crf
    total_project_cost = net_bill + annualized_battery_cost + annualized_fixed_pv
    
    usable_discharge = capacity_kwh * (assumptions.max_soc_fraction - assumptions.min_soc_fraction) * eta_dis
    avg_daily_load = float(load.sum()) / (T / 24.0)
    days_of_autonomy = usable_discharge / avg_daily_load if avg_daily_load > 0 else 0.0
    
    # Recharge energy needed from surplus
    recharge_energy_kwh = capacity_kwh * (assumptions.max_soc_fraction - assumptions.min_soc_fraction) / eta_ch
    # We will compute recharge days against average daily PV surplus of the 0-battery case
    
    summary = SizingResult(
        battery_capacity_kwh=capacity_kwh,
        battery_power_kw=power_kw,
        grid_import_kwh=grid_import_kwh,
        grid_export_kwh=grid_export_kwh,
        grid_import_cost_rs=grid_import_cost,
        grid_export_revenue_rs=grid_export_revenue,
        net_energy_bill_rs=net_bill,
        pv_to_load_kwh=float(p_pv_load.sum()),
        pv_to_battery_kwh=float(p_pv_bat.sum()),
        battery_to_load_kwh=float(p_bat_load.sum()),
        curtailed_solar_kwh=0.0,
        usable_discharge_energy_kwh=usable_discharge,
        days_of_autonomy=days_of_autonomy,
        hours_of_autonomy=days_of_autonomy * 24.0,
        equivalent_recharge_days=0.0,  # filled after baseline is known
        annualized_battery_cost_rs=annualized_battery_cost,
        annualized_fixed_pv_cost_rs=annualized_fixed_pv,
        annualized_total_project_cost_rs=total_project_cost,
        annual_savings_vs_no_battery_rs=0.0,
        simple_payback_years=None,
    )
    dispatch_dict = {
        "pv_to_load": p_pv_load,
        "pv_to_battery": p_pv_bat,
        "pv_to_grid": p_pv_grid,
        "battery_to_load": p_bat_load,
        "grid_to_load": p_grid_load,
        "soc": soc,
    }
    return summary, dispatch_dict


def write_line_chart_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], list[float], str]],
    highlight: tuple[float, float, str] | None = None,
) -> None:
    width, height = 980, 600
    left, right, top, bottom = 110, 40, 60, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [x for _, xs, _, _ in series for x in xs]
    y_values = [y for _, _, ys, _ in series for y in ys]
    x_min, x_max = 0.0, max(x_values) * 1.05 if max(x_values) else 1.0
    y_min, y_max = min(0.0, min(y_values) * 1.1 if min(y_values) < 0 else 0.0), max(y_values) * 1.12 if max(y_values) else 1.0

    def sx(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_width

    def sy(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; fill: #1f2937; } .grid { stroke: #e5e7eb; stroke-width: 1; } .axis { stroke: #374151; stroke-width: 1.4; } .tick { font-size: 13px; fill: #4b5563; } .title { font-size: 20px; font-weight: 700; fill: #111827; } .label { font-size: 15px; font-weight: 600; fill: #374151; }</style>',
        f'<text x="{width / 2}" y="32" class="title" text-anchor="middle">{escape(title)}</text>',
    ]
    for tick in range(6):
        v = y_min + (y_max - y_min) * tick / 5
        y = sy(v)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for tick in range(6):
        v = x_min + (x_max - x_min) * tick / 5
        x = sx(v)
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{height - bottom}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - bottom + 24}" class="tick" text-anchor="middle">{v:.1f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" class="axis"/>',
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" class="axis"/>',
            f'<text x="{width / 2}" y="{height - 35}" class="label" text-anchor="middle">{escape(x_label)}</text>',
            f'<text x="28" y="{height / 2}" class="label" text-anchor="middle" transform="rotate(-90 28 {height / 2})">{escape(y_label)}</text>',
        ]
    )
    for index, (name, xs, ys, color) in enumerate(series):
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
        legend_y = top + 8 + index * 24
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6" points="{pts}"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.8" fill="{color}"/>')
        parts.append(f'<line x1="{width - right - 220}" x2="{width - right - 195}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right - 188}" y="{legend_y + 4}" class="tick">{escape(name)}</text>')
    if highlight:
        x, y, label = highlight
        px, py = sx(x), sy(y)
        lx = min(px + 24, width - right - 220)
        ly = max(py - 36, top + 24)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="#dc2626"/>')
        parts.append(f'<line x1="{px:.2f}" x2="{lx:.2f}" y1="{py:.2f}" y2="{ly:.2f}" stroke="#dc2626" stroke-width="1.5"/>')
        for r_idx, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{lx + 6:.2f}" y="{ly + r_idx * 17:.2f}" class="tick" fill="#dc2626" font-weight="600">{escape(line)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_dispatch_chart_svg(
    path: Path,
    title: str,
    pv: np.ndarray,
    load: np.ndarray,
    dispatch: dict[str, np.ndarray],
    start_hour: int = 24 * 70,  # e.g., mid-March sunny week
    hours_to_plot: int = 168,   # 7 days
) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 60, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    
    t_range = range(start_hour, start_hour + hours_to_plot)
    p_pv = [float(pv[t]) for t in t_range]
    p_load = [float(load[t]) for t in t_range]
    p_bat_ch = [float(dispatch["pv_to_battery"][t]) for t in t_range]
    p_bat_dis = [float(dispatch["battery_to_load"][t]) for t in t_range]
    p_grid_imp = [float(dispatch["grid_to_load"][t]) for t in t_range]
    p_grid_exp = [float(dispatch["pv_to_grid"][t]) for t in t_range]
    soc = [float(dispatch["soc"][t]) for t in t_range]
    
    max_power = max(max(p_pv), max(p_load), max(p_bat_ch), max(p_bat_dis), max(p_grid_imp), max(p_grid_exp)) * 1.15
    max_soc = max(soc) * 1.15 if max(soc) > 0 else 1.0
    
    def sx(h_idx: int) -> float:
        return left + h_idx / (hours_to_plot - 1) * plot_width
        
    def sy(p: float) -> float:
        return top + (max_power - p) / max_power * plot_height

    def sy_soc(s: float) -> float:
        return top + (max_soc - s) / max_soc * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; fill: #1f2937; } .grid { stroke: #f3f4f6; stroke-width: 1; } .axis { stroke: #374151; stroke-width: 1.4; } .tick { font-size: 12px; fill: #4b5563; } .title { font-size: 20px; font-weight: 700; fill: #111827; } .label { font-size: 14px; font-weight: 600; fill: #374151; }</style>',
        f'<text x="{width / 2}" y="32" class="title" text-anchor="middle">{escape(title)}</text>',
    ]
    # Grid lines
    for tick in range(6):
        v = max_power * tick / 5
        y = sy(v)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" class="tick" text-anchor="end">{v:.0f}</text>')
    for day in range(8):
        x = sx(day * 24)
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{height - bottom}" class="axis" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - bottom + 20}" class="tick" text-anchor="middle">Day {day+1}</text>')

    parts.extend([
        f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" class="axis"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" class="axis"/>',
        f'<text x="{width / 2}" y="{height - 35}" class="label" text-anchor="middle">Time (7 Days / 168 Hours)</text>',
        f'<text x="25" y="{height / 2}" class="label" text-anchor="middle" transform="rotate(-90 25 {height / 2})">Power (kW)</text>',
    ])
    
    # Series
    traces = [
        ("Solar PV", p_pv, "#f59e0b", 2.2),
        ("Load Demand", p_load, "#111827", 2.0),
        ("Battery Charge", p_bat_ch, "#10b981", 1.8),
        ("Battery Discharge", p_bat_dis, "#6366f1", 2.0),
        ("Grid Import", p_grid_imp, "#ef4444", 1.8),
        ("Grid Export", p_grid_exp, "#06b6d4", 1.6),
    ]
    for idx, (name, series, color, stroke_w) in enumerate(traces):
        pts = " ".join(f"{sx(i):.2f},{sy(val):.2f}" for i, val in enumerate(series))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_w}" points="{pts}"/>')
        
        # Legend item
        leg_x = left + 10 + (idx % 3) * 220
        leg_y = height - 65 + (idx // 3) * 20
        parts.append(f'<line x1="{leg_x}" x2="{leg_x + 22}" y1="{leg_y}" y2="{leg_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{leg_x + 28}" y="{leg_y + 4}" class="tick">{escape(name)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run_continuous_optimization(
    irradiance_csv: Path,
    load_csv: Path,
    output_dir: Path,
    candidate_capacities: list[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assumptions = ProjectAssumptions()
    
    pv_monthly = read_pvsyst_monthly(irradiance_csv)
    daily_load = read_load_profile(load_csv)
    pv, load, c_imp, c_exp = build_annual_timeseries(pv_monthly, daily_load)
    
    total_pv_kwh = float(pv.sum())
    total_load_kwh = float(load.sum())
    days_in_year = len(pv) / 24.0
    avg_daily_load_kwh = total_load_kwh / days_in_year
    
    print(f"Loaded annual PV: {total_pv_kwh:,.1f} kWh | Annual Load: {total_load_kwh:,.1f} kWh | Daily Load: {avg_daily_load_kwh:,.1f} kWh/day")
    print(f"Dynamic ToD Import Tariff: Rs 6.50 (night), Rs 8.00 (day), Rs 10.40 (peak evening)")
    print(f"Variable Export Tariff: Rs 2.50 (midday solar), Rs 3.20 (morning), Rs 3.80 (afternoon), Rs 5.50 (evening)")
    
    # 1. Baseline: 0 kWh
    no_bess, no_bess_dispatch = solve_dispatch_lp(pv, load, c_imp, c_exp, 0.0, 0.0, assumptions)
    avg_daily_pv_surplus_kwh = no_bess.grid_export_kwh / days_in_year
    
    # 2. Continuous sweep from 1 to 1000 kWh
    results: list[SizingResult] = [no_bess]
    best_dispatch = None
    best_capacity = 0.0
    min_project_cost = no_bess.annualized_total_project_cost_rs
    
    eta_ch = math.sqrt(assumptions.battery_efficiency)
    eta_dis = math.sqrt(assumptions.battery_efficiency)
    
    print("\nRunning continuous LP optimization sweep across 1 to 1,000 kWh...")
    for cap in candidate_capacities:
        power = min(1000.0, cap / assumptions.storage_duration_hours) if cap > 0 else 0.0
        summary, dispatch = solve_dispatch_lp(pv, load, c_imp, c_exp, cap, power, assumptions)
        
        # Calculate savings and payback vs no-battery
        annual_savings = no_bess.net_energy_bill_rs - summary.net_energy_bill_rs
        summary.annual_savings_vs_no_battery_rs = annual_savings
        
        capex = cap * assumptions.battery_cost_rs_per_kwh + power * (assumptions.inverter_cost_rs_per_mw / 1000.0)
        summary.simple_payback_years = capex / annual_savings if annual_savings > 0 else None
        
        # Equivalent recharge days
        recharge_energy = cap * (assumptions.max_soc_fraction - assumptions.min_soc_fraction) / eta_ch
        summary.equivalent_recharge_days = recharge_energy / avg_daily_pv_surplus_kwh if avg_daily_pv_surplus_kwh > 0 else 0.0
        
        results.append(summary)
        if summary.annualized_total_project_cost_rs < min_project_cost:
            min_project_cost = summary.annualized_total_project_cost_rs
            best_capacity = cap
            best_dispatch = dispatch
            
    # Export full sizing curve CSV
    csv_path = output_dir / "milp_continuous_sizing_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
            
    # Export optimal / representative dispatch timeseries CSV
    rep_cap = best_capacity if best_capacity > 0 else 500.0
    rep_summary, rep_dispatch = solve_dispatch_lp(
        pv, load, c_imp, c_exp, rep_cap, min(1000.0, rep_cap / assumptions.storage_duration_hours), assumptions
    )
    dispatch_csv_path = output_dir / "milp_optimal_dispatch_8760h.csv"
    with dispatch_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["hour_index", "pv_available_kwh", "load_kwh", "import_tariff_rs", "export_tariff_rs",
                      "pv_to_load_kwh", "pv_to_battery_kwh", "pv_to_grid_kwh", "battery_to_load_kwh", "grid_to_load_kwh", "soc_kwh"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(len(pv)):
            writer.writerow({
                "hour_index": t,
                "pv_available_kwh": pv[t],
                "load_kwh": load[t],
                "import_tariff_rs": c_imp[t],
                "export_tariff_rs": c_exp[t],
                "pv_to_load_kwh": rep_dispatch["pv_to_load"][t],
                "pv_to_battery_kwh": rep_dispatch["pv_to_battery"][t],
                "pv_to_grid_kwh": rep_dispatch["pv_to_grid"][t],
                "battery_to_load_kwh": rep_dispatch["battery_to_load"][t],
                "grid_to_load_kwh": rep_dispatch["grid_to_load"][t],
                "soc_kwh": rep_dispatch["soc"][t],
            })

    # Generate Charts
    caps = [r.battery_capacity_kwh for r in results]
    total_cost_m = [r.annualized_total_project_cost_rs / 1_000_000 for r in results]
    bill_m = [r.net_energy_bill_rs / 1_000_000 for r in results]
    bess_m = [r.annualized_battery_cost_rs / 1_000_000 for r in results]
    exp_rev_m = [r.grid_export_revenue_rs / 1_000_000 for r in results]
    imp_cost_m = [r.grid_import_cost_rs / 1_000_000 for r in results]
    autonomy_days = [r.days_of_autonomy for r in results]
    recharge_days = [r.equivalent_recharge_days for r in results]

    optimum = min(results, key=lambda x: x.annualized_total_project_cost_rs)
    nonzero_results = [r for r in results if r.battery_capacity_kwh > 0]
    best_nonzero = min(nonzero_results, key=lambda x: x.annualized_total_project_cost_rs)

    # 1. Cost vs Capacity Continuous Chart
    chart_cap_path = output_dir / "milp_cost_vs_capacity_continuous.svg"
    write_line_chart_svg(
        chart_cap_path,
        "Annualized Total Project Cost vs Battery Capacity (Continuous 1–1000 kWh)",
        "Battery Capacity (kWh)",
        "Annualized Total Project Cost (Rs million/year)",
        [("Total Project Cost", caps, total_cost_m, "#059669")],
        (
            optimum.battery_capacity_kwh,
            optimum.annualized_total_project_cost_rs / 1_000_000,
            f"Optimum: {optimum.battery_capacity_kwh:.0f} kWh\nRs {optimum.annualized_total_project_cost_rs / 1_000_000:.2f} M/yr",
        ),
    )

    # 2. Cost Components vs Capacity
    chart_comp_cap_path = output_dir / "milp_cost_components_vs_capacity.svg"
    write_line_chart_svg(
        chart_comp_cap_path,
        "Annual Cost Components vs Battery Capacity (Continuous 1–1000 kWh)",
        "Battery Capacity (kWh)",
        "Cost (Rs million/year)",
        [
            ("Net Grid Bill (Import - Export)", caps, bill_m, "#2563eb"),
            ("Gross Import Cost", caps, imp_cost_m, "#dc2626"),
            ("Export Revenue", caps, exp_rev_m, "#06b6d4"),
            ("Annualized BESS Cost", caps, bess_m, "#d97706"),
            ("Total Project Cost", caps, total_cost_m, "#059669"),
        ],
    )

    # 3. Cost vs Days of Autonomy
    chart_autonomy_path = output_dir / "milp_cost_vs_days_of_autonomy.svg"
    write_line_chart_svg(
        chart_autonomy_path,
        "Annualized Project Cost vs Days of Autonomy (Continuous LP)",
        "Days of Autonomy (usable storage from max to min SOC)",
        "Annualized Total Project Cost (Rs million/year)",
        [("Total Project Cost", autonomy_days, total_cost_m, "#059669")],
        (
            optimum.days_of_autonomy,
            optimum.annualized_total_project_cost_rs / 1_000_000,
            f"Optimum: {optimum.days_of_autonomy:.2f} days\nRs {optimum.annualized_total_project_cost_rs / 1_000_000:.2f} M/yr",
        ),
    )

    # Battery capacity vs days of autonomy
    chart_capacity_autonomy_path = output_dir / "milp_battery_capacity_vs_days_of_autonomy.svg"
    write_line_chart_svg(
        chart_capacity_autonomy_path,
        "Battery Capacity vs Days of Autonomy",
        "Battery Capacity (kWh)",
        "Days of Autonomy",
        [("Days of Autonomy", caps, autonomy_days, "#059669")],
        (
            optimum.battery_capacity_kwh,
            optimum.days_of_autonomy,
            f"Optimum: {optimum.battery_capacity_kwh:.0f} kWh\n{optimum.days_of_autonomy:.2f} days",
        ),
    )

    # 4. Cost Components vs Days of Autonomy
    chart_comp_autonomy_path = output_dir / "milp_cost_components_vs_autonomy.svg"
    write_line_chart_svg(
        chart_comp_autonomy_path,
        "Annual Cost Components vs Days of Autonomy (Continuous LP)",
        "Days of Autonomy (usable storage from max to min SOC)",
        "Cost (Rs million/year)",
        [
            ("Net Grid Bill", autonomy_days, bill_m, "#2563eb"),
            ("Annualized BESS Cost", autonomy_days, bess_m, "#d97706"),
            ("Total Project Cost", autonomy_days, total_cost_m, "#059669"),
        ],
    )

    # 5. Cost vs Equivalent Recharge Days
    chart_recharge_path = output_dir / "milp_cost_vs_recharge_days.svg"
    write_line_chart_svg(
        chart_recharge_path,
        "Annualized Project Cost vs Equivalent Recharge Days (Continuous LP)",
        "Equivalent Recharge Days from Min to Max SOC",
        "Annualized Total Project Cost (Rs million/year)",
        [("Total Project Cost", recharge_days, total_cost_m, "#059669")],
        (
            optimum.equivalent_recharge_days,
            optimum.annualized_total_project_cost_rs / 1_000_000,
            f"Optimum: {optimum.equivalent_recharge_days:.2f} days\nRs {optimum.annualized_total_project_cost_rs / 1_000_000:.2f} M/yr",
        ),
    )

    # PV capacity vs recharge days for the best non-zero battery size
    pv_capacity_candidates_mw = [step / 10.0 for step in range(5, 21)]
    recharge_energy_kwh = (
        best_nonzero.battery_capacity_kwh
        * (assumptions.max_soc_fraction - assumptions.min_soc_fraction)
        / eta_ch
    )
    pv_recharge_days = []
    for pv_capacity_mw in pv_capacity_candidates_mw:
        scaled_pv = pv * (pv_capacity_mw / assumptions.pv_rating_mw)
        daily_pv_surplus_kwh = float(np.maximum(scaled_pv - load, 0.0).sum()) / days_in_year
        pv_recharge_days.append(
            recharge_energy_kwh / daily_pv_surplus_kwh if daily_pv_surplus_kwh > 0 else 0.0
        )

    chart_pv_recharge_path = output_dir / "milp_pv_capacity_vs_recharge_days.svg"
    write_line_chart_svg(
        chart_pv_recharge_path,
        f"PV Capacity vs Recharge Days ({best_nonzero.battery_capacity_kwh:.0f} kWh Battery)",
        "PV Capacity (MW)",
        "Equivalent Recharge Days",
        [("Recharge Days", pv_capacity_candidates_mw, pv_recharge_days, "#2563eb")],
        (
            assumptions.pv_rating_mw,
            best_nonzero.equivalent_recharge_days,
            f"Current: {assumptions.pv_rating_mw:.1f} MW\n{best_nonzero.equivalent_recharge_days:.2f} days",
        ),
    )

    # 6. Sample 7-day hourly dispatch chart
    chart_dispatch_path = output_dir / "milp_dispatch_sample_week.svg"
    write_dispatch_chart_svg(
        chart_dispatch_path,
        f"MILP Optimal Hourly Dispatch Sample (Representative {rep_cap:.0f} kWh BESS)",
        pv, load, rep_dispatch,
        start_hour=24 * 75,  # mid March
        hours_to_plot=168,
    )

    # Summary JSON
    summary_path = output_dir / "milp_optimization_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "framework": "Continuous LP/MILP Battery Sizing & Hourly Optimal Dispatch",
                "assumptions": asdict(assumptions),
                "tariffs": {
                    "import_tod": {
                        "peak_18_to_22": 10.40,
                        "normal_06_to_18": 8.00,
                        "off_peak_22_to_06": 6.50
                    },
                    "export_variable": {
                        "midday_solar_10_to_16": 2.50,
                        "morning_shoulder_06_to_10": 3.20,
                        "afternoon_shoulder_16_to_18": 3.80,
                        "evening_peak_18_to_22": 5.50,
                        "night_base_22_to_06": 2.50
                    }
                },
                "annual_totals": {
                    "solar_pv_generation_kwh": total_pv_kwh,
                    "load_consumption_kwh": total_load_kwh,
                    "avg_daily_load_kwh": avg_daily_load_kwh,
                    "avg_daily_pv_surplus_kwh": avg_daily_pv_surplus_kwh,
                },
                "no_battery_case": asdict(no_bess),
                "overall_optimal_case": asdict(optimum),
                "best_nonzero_bess_case": asdict(best_nonzero),
                "evaluated_capacities_count": len(results),
            },
            f,
            indent=2,
        )

    print("\n" + "="*70)
    print("CONTINUOUS OPTIMIZATION SUMMARY (1 to 1,000 kWh)")
    print("="*70)
    print(f"No Battery (0 kWh):")
    print(f"  Net Grid Bill: Rs {no_bess.net_energy_bill_rs:,.0f}/yr (Imports: Rs {no_bess.grid_import_cost_rs:,.0f}, Export Rev: Rs {no_bess.grid_export_revenue_rs:,.0f})")
    print(f"  Total Project Cost: Rs {no_bess.annualized_total_project_cost_rs:,.0f}/yr")
    print(f"\nOptimal Case Overall:")
    print(f"  Capacity: {optimum.battery_capacity_kwh:.1f} kWh | Autonomy: {optimum.days_of_autonomy:.2f} days ({optimum.hours_of_autonomy:.1f} h)")
    print(f"  Total Project Cost: Rs {optimum.annualized_total_project_cost_rs:,.0f}/yr")
    print(f"\nBest Performing Non-Zero Storage System:")
    print(f"  Capacity: {best_nonzero.battery_capacity_kwh:.1f} kWh ({best_nonzero.battery_power_kw:.1f} kW PCS)")
    print(f"  Days of Autonomy: {best_nonzero.days_of_autonomy:.3f} days ({best_nonzero.hours_of_autonomy:.1f} hours)")
    print(f"  Equivalent Recharge Days: {best_nonzero.equivalent_recharge_days:.3f} days")
    print(f"  Net Energy Bill: Rs {best_nonzero.net_energy_bill_rs:,.0f}/yr (Annual Savings: Rs {best_nonzero.annual_savings_vs_no_battery_rs:,.0f}/yr)")
    print(f"  Annualized BESS Cost: Rs {best_nonzero.annualized_battery_cost_rs:,.0f}/yr")
    print(f"  Total Project Cost: Rs {best_nonzero.annualized_total_project_cost_rs:,.0f}/yr")
    if best_nonzero.simple_payback_years is not None:
        print(f"  Simple Payback: {best_nonzero.simple_payback_years:.2f} years")

    print("\nGenerated output files:")
    print(f"  - {csv_path}")
    print(f"  - {dispatch_csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {chart_cap_path}")
    print(f"  - {chart_comp_cap_path}")
    print(f"  - {chart_autonomy_path}")
    print(f"  - {chart_capacity_autonomy_path}")
    print(f"  - {chart_comp_autonomy_path}")
    print(f"  - {chart_recharge_path}")
    print(f"  - {chart_pv_recharge_path}")
    print(f"  - {chart_dispatch_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Battery Sizing and MILP Optimal Dispatch with Dynamic MP Tariffs.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--irradiance-csv",
        type=Path,
        default=script_dir / "data" / "irradiance_iit_indore.csv",
    )
    parser.add_argument(
        "--load-csv",
        type=Path,
        default=script_dir / "data" / "load_profile_new.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
    )
    args = parser.parse_args()

    # Continuous sizing sweep points: dense from 1 to 1000 kWh
    dense_capacities = [
        1.0,5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0,
        350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0,
        800.0, 850.0, 900.0, 950.0, 1000.0
    ]
    run_continuous_optimization(args.irradiance_csv, args.load_csv, args.output_dir, dense_capacities)


if __name__ == "__main__":
    main()
