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


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@dataclass(frozen=True)
class ProjectAssumptions:
    pv_rating_mw: float = 1.0
    import_tariff_rs_per_kwh: float = 10.0
    export_credit_rs_per_kwh: float = 0.5
    pv_cost_rs_per_mw: float = 40_000_000.0
    battery_cost_rs_per_kwh: float = 9_500.0
    inverter_cost_rs_per_mw: float = 320_000.0
    land_and_installation_rs_per_mw: float = 5_000_000.0
    battery_efficiency: float = 0.92
    initial_soc_fraction: float = 0.50
    min_soc_fraction: float = 0.20
    max_soc_fraction: float = 0.90
    project_years: int = 15
    discount_rate: float = 0.08


@dataclass
class DispatchSummary:
    battery_capacity_kwh: float
    battery_power_kw: float
    grid_import_kwh: float
    grid_export_kwh: float
    pv_to_load_kwh: float
    pv_to_battery_kwh: float
    battery_to_load_kwh: float
    unmet_load_kwh: float
    final_soc_kwh: float
    energy_bill_rs: float
    annualized_battery_cost_rs: float
    annualized_total_project_cost_rs: float
    annual_savings_vs_no_battery_rs: float
    simple_payback_years: float | None


def read_pvsyst_monthly(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = list(csv.reader(file))

    header: list[str] | None = None
    data: dict[str, dict[str, float]] = {}
    for row in rows:
        if len(row) > 2 and row[1] == "GlobHor":
            header = ["Month", *row[1:]]
            continue
        if header and row and row[0] in MONTHS:
            values = {}
            for name, value in zip(header[1:], row[1:]):
                values[name] = float(value)
            data[row[0]] = values

    missing = [month for month in MONTHS if month not in data]
    if missing:
        raise ValueError(f"Missing month rows in PVsyst CSV: {', '.join(missing)}")
    return data


def read_load_profile(path: Path) -> list[float]:
    hourly_load = [None] * 24
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            start, end = number_pair(row["Time"])
            low, high = number_pair(row["Load (kW)"])
            load_kw = (low + high) / 2.0
            for hour in range(start, end):
                hourly_load[hour % 24] = load_kw

    if any(value is None for value in hourly_load):
        known = [value for value in hourly_load if value is not None]
        fallback = sum(known) / len(known) if known else 0.0
        hourly_load = [fallback if value is None else value for value in hourly_load]
    return [float(value) for value in hourly_load]


def number_pair(text: str) -> tuple[int, int]:
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if len(numbers) < 2:
        raise ValueError(f"Expected a numeric range, got: {text}")
    return numbers[0], numbers[1]


def build_hourly_profile(
    pv_monthly: dict[str, dict[str, float]],
    daily_load_kw: list[float],
    year: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    hour_index = 0
    for month_number, month_name in enumerate(MONTHS, start=1):
        days = calendar.monthrange(year, month_number)[1]
        month_weights = []
        for _day in range(days):
            for hour in range(24):
                # Simple daylight production shape. The monthly PVsyst E_Grid total sets the scale.
                daylight_weight = max(0.0, math.sin(math.pi * (hour - 6) / 12.0))
                month_weights.append(daylight_weight)

        monthly_pv_kwh = pv_monthly[month_name]["E_Grid"]
        scale = monthly_pv_kwh / sum(month_weights) if sum(month_weights) else 0.0

        for day in range(1, days + 1):
            for hour in range(24):
                weight = month_weights[(day - 1) * 24 + hour]
                rows.append(
                    {
                        "hour_index": hour_index,
                        "month": month_name,
                        "day": day,
                        "hour": hour,
                        "pv_available_kwh": weight * scale,
                        "load_kwh": daily_load_kw[hour],
                    }
                )
                hour_index += 1
    return rows


def simulate_bess(
    hourly_rows: list[dict[str, float | int | str]],
    capacity_kwh: float,
    power_kw: float,
    assumptions: ProjectAssumptions,
) -> tuple[DispatchSummary, list[dict[str, float | int | str]]]:
    charge_efficiency = math.sqrt(assumptions.battery_efficiency)
    discharge_efficiency = math.sqrt(assumptions.battery_efficiency)
    min_soc = capacity_kwh * assumptions.min_soc_fraction
    max_soc = capacity_kwh * assumptions.max_soc_fraction
    soc = capacity_kwh * assumptions.initial_soc_fraction

    grid_import = 0.0
    grid_export = 0.0
    pv_to_load = 0.0
    pv_to_battery = 0.0
    battery_to_load = 0.0
    unmet_load = 0.0
    dispatch_rows: list[dict[str, float | int | str]] = []

    for row in hourly_rows:
        pv = float(row["pv_available_kwh"])
        load = float(row["load_kwh"])

        direct_pv = min(pv, load)
        surplus = pv - direct_pv
        deficit = load - direct_pv

        charge_room_from_meter = max((max_soc - soc) / charge_efficiency, 0.0) if capacity_kwh else 0.0
        charge = min(surplus, power_kw, charge_room_from_meter)
        soc += charge * charge_efficiency
        surplus -= charge

        discharge_available_to_meter = max((soc - min_soc) * discharge_efficiency, 0.0) if capacity_kwh else 0.0
        discharge = min(deficit, power_kw, discharge_available_to_meter)
        soc -= discharge / discharge_efficiency if discharge_efficiency else 0.0
        deficit -= discharge

        export = surplus
        grid_buy = deficit

        pv_to_load += direct_pv
        pv_to_battery += charge
        battery_to_load += discharge
        grid_export += export
        grid_import += grid_buy

        dispatch_row = dict(row)
        dispatch_row.update(
            {
                "pv_to_load_kwh": direct_pv,
                "pv_to_battery_kwh": charge,
                "battery_to_load_kwh": discharge,
                "grid_import_kwh": grid_buy,
                "grid_export_kwh": export,
                "soc_kwh": soc,
            }
        )
        dispatch_rows.append(dispatch_row)

    energy_bill = (
        grid_import * assumptions.import_tariff_rs_per_kwh
        - grid_export * assumptions.export_credit_rs_per_kwh
    )
    battery_capex = (
        capacity_kwh * assumptions.battery_cost_rs_per_kwh
        + (power_kw / 1000.0) * assumptions.inverter_cost_rs_per_mw
    )
    annualized_battery_cost = battery_capex * capital_recovery_factor(
        assumptions.discount_rate,
        assumptions.project_years,
    )
    fixed_pv_capex = assumptions.pv_rating_mw * (
        assumptions.pv_cost_rs_per_mw + assumptions.land_and_installation_rs_per_mw
    )
    annualized_fixed_pv_cost = fixed_pv_capex * capital_recovery_factor(
        assumptions.discount_rate,
        assumptions.project_years,
    )
    total_project_cost = energy_bill + annualized_battery_cost + annualized_fixed_pv_cost

    summary = DispatchSummary(
        battery_capacity_kwh=capacity_kwh,
        battery_power_kw=power_kw,
        grid_import_kwh=grid_import,
        grid_export_kwh=grid_export,
        pv_to_load_kwh=pv_to_load,
        pv_to_battery_kwh=pv_to_battery,
        battery_to_load_kwh=battery_to_load,
        unmet_load_kwh=unmet_load,
        final_soc_kwh=soc,
        energy_bill_rs=energy_bill,
        annualized_battery_cost_rs=annualized_battery_cost,
        annualized_total_project_cost_rs=total_project_cost,
        annual_savings_vs_no_battery_rs=0.0,
        simple_payback_years=None,
    )
    return summary, dispatch_rows


def capital_recovery_factor(rate: float, years: int) -> float:
    if rate == 0:
        return 1.0 / years
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def write_line_chart_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], list[float], str]],
    highlight: tuple[float, float, str] | None = None,
) -> None:
    """Write a dependency-free SVG line chart with labeled axes and legend."""
    width, height = 960, 590
    left, right, top, bottom = 105, 38, 55, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [x for _, xs, _, _ in series for x in xs]
    y_values = [y for _, _, ys, _ in series for y in ys]
    x_min, x_max = 0.0, max(x_values) * 1.06 if max(x_values) else 1.0
    y_min, y_max = 0.0, max(y_values) * 1.12 if max(y_values) else 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, sans-serif; fill: #1f2937; } .grid { stroke: #d1d5db; stroke-width: 1; } .axis { stroke: #374151; stroke-width: 1.3; } .tick { font-size: 13px; } .title { font-size: 20px; font-weight: 600; } .label { font-size: 15px; }</style>',
        f'<text x="{width / 2}" y="30" class="title" text-anchor="middle">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" class="tick" text-anchor="end">{value:.1f}</text>')
    for tick in range(6):
        value = x_min + (x_max - x_min) * tick / 5
        x = sx(value)
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{height - bottom}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - bottom + 24}" class="tick" text-anchor="middle">{value:.2f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" class="axis"/>',
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" class="axis"/>',
            f'<text x="{width / 2}" y="{height - 32}" class="label" text-anchor="middle">{escape(x_label)}</text>',
            f'<text x="25" y="{height / 2}" class="label" text-anchor="middle" transform="rotate(-90 25 {height / 2})">{escape(y_label)}</text>',
        ]
    )
    for index, (name, xs, ys, color) in enumerate(series):
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
        legend_y = top + 4 + index * 24
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="{color}"/>')
        parts.append(f'<line x1="{width - right - 205}" x2="{width - right - 180}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right - 173}" y="{legend_y + 5}" class="tick">{escape(name)}</text>')
    if highlight:
        x, y, label = highlight
        point_x, point_y = sx(x), sy(y)
        label_x = min(point_x + 24, width - right - 180)
        label_y = max(point_y - 32, top + 24)
        parts.append(f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="8" fill="#c0392b"/>')
        parts.append(f'<line x1="{point_x:.2f}" x2="{label_x:.2f}" y1="{point_y:.2f}" y2="{label_y:.2f}" stroke="#c0392b" stroke-width="1.5"/>')
        for row, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{label_x + 5:.2f}" y="{label_y + row * 18:.2f}" class="tick" fill="#c0392b">{escape(line)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def create_recharge_days_outputs(
    summaries: list[DispatchSummary],
    no_battery: DispatchSummary,
    assumptions: ProjectAssumptions,
    days_in_model: float,
    output_dir: Path,
) -> tuple[dict[str, float], Path, Path, Path]:
    """Export recharge-day sensitivity data and two cost-optimization charts.

    Recharge days are the PV-surplus days needed to charge the BESS from its
    minimum SOC to its maximum SOC, using the no-BESS average daily PV export.
    """
    charge_efficiency = math.sqrt(assumptions.battery_efficiency)
    average_daily_pv_surplus_kwh = no_battery.grid_export_kwh / days_in_model
    annualized_fixed_pv_cost_rs = assumptions.pv_rating_mw * (
        assumptions.pv_cost_rs_per_mw + assumptions.land_and_installation_rs_per_mw
    ) * capital_recovery_factor(assumptions.discount_rate, assumptions.project_years)

    plot_rows: list[dict[str, float]] = []
    for summary in sorted(summaries, key=lambda item: item.battery_capacity_kwh):
        recharge_energy_kwh = (
            summary.battery_capacity_kwh
            * (assumptions.max_soc_fraction - assumptions.min_soc_fraction)
            / charge_efficiency
        )
        recharge_days = (
            recharge_energy_kwh / average_daily_pv_surplus_kwh
            if average_daily_pv_surplus_kwh > 0
            else 0.0
        )
        plot_rows.append(
            {
                "battery_capacity_kwh": summary.battery_capacity_kwh,
                "battery_power_kw": summary.battery_power_kw,
                "full_recharge_energy_kwh": recharge_energy_kwh,
                "equivalent_recharge_days": recharge_days,
                "annual_energy_bill_rs": summary.energy_bill_rs,
                "annualized_battery_cost_rs": summary.annualized_battery_cost_rs,
                "annualized_fixed_pv_cost_rs": annualized_fixed_pv_cost_rs,
                "annualized_total_project_cost_rs": summary.annualized_total_project_cost_rs,
                "annual_savings_vs_no_battery_rs": summary.annual_savings_vs_no_battery_rs,
            }
        )

    data_path = output_dir / "bess_recharge_days_comparison.csv"
    with data_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    optimum = min(plot_rows, key=lambda item: item["annualized_total_project_cost_rs"])
    nonzero_rows = [row for row in plot_rows if row["battery_capacity_kwh"] > 0]
    best_nonzero = min(nonzero_rows, key=lambda item: item["annualized_total_project_cost_rs"])

    recharge_days = [row["equivalent_recharge_days"] for row in plot_rows]
    total_cost_million = [row["annualized_total_project_cost_rs"] / 1_000_000 for row in plot_rows]
    bill_million = [row["annual_energy_bill_rs"] / 1_000_000 for row in plot_rows]
    battery_cost_million = [row["annualized_battery_cost_rs"] / 1_000_000 for row in plot_rows]

    chart_total_path = output_dir / "cost_vs_recharge_days.svg"
    write_line_chart_svg(
        chart_total_path,
        "Annualized Project Cost vs Equivalent Recharge Days",
        "Equivalent recharge days from minimum to maximum SOC",
        "Annualized total project cost (Rs million/year)",
        [("Total project cost", recharge_days, total_cost_million, "#0b6e4f")],
        (
            optimum["equivalent_recharge_days"],
            optimum["annualized_total_project_cost_rs"] / 1_000_000,
            f"Optimum: {optimum['equivalent_recharge_days']:.2f} days\nRs {optimum['annualized_total_project_cost_rs'] / 1_000_000:.2f} million/year",
        ),
    )

    chart_components_path = output_dir / "cost_components_vs_recharge_days.svg"
    write_line_chart_svg(
        chart_components_path,
        "Annual Cost Components vs Equivalent Recharge Days",
        "Equivalent recharge days from minimum to maximum SOC",
        "Cost (Rs million/year)",
        [
            ("Annual grid bill", recharge_days, bill_million, "#1f77b4"),
            ("Annualized BESS cost", recharge_days, battery_cost_million, "#d97706"),
            ("Total project cost", recharge_days, total_cost_million, "#0b6e4f"),
        ],
    )

    optimum["average_daily_pv_surplus_kwh"] = average_daily_pv_surplus_kwh
    optimum["lowest_cost_nonzero_bess_capacity_kwh"] = best_nonzero["battery_capacity_kwh"]
    optimum["lowest_cost_nonzero_bess_recharge_days"] = best_nonzero["equivalent_recharge_days"]
    optimum["lowest_cost_nonzero_bess_project_cost_rs"] = best_nonzero["annualized_total_project_cost_rs"]
    return optimum, data_path, chart_total_path, chart_components_path


def create_autonomy_days_outputs(
    summaries: list[DispatchSummary],
    total_load_kwh: float,
    assumptions: ProjectAssumptions,
    days_in_model: float,
    output_dir: Path,
) -> tuple[dict[str, float], Path, Path, Path]:
    """Export autonomy-day sensitivity data and two cost-optimization charts.

    Days of autonomy represent the duration (in days) the BESS can sustain
    the average daily load when discharging from maximum SOC to minimum SOC,
    accounting for discharge efficiency.
    """
    discharge_efficiency = math.sqrt(assumptions.battery_efficiency)
    average_daily_load_kwh = total_load_kwh / days_in_model
    annualized_fixed_pv_cost_rs = assumptions.pv_rating_mw * (
        assumptions.pv_cost_rs_per_mw + assumptions.land_and_installation_rs_per_mw
    ) * capital_recovery_factor(assumptions.discount_rate, assumptions.project_years)

    plot_rows: list[dict[str, float]] = []
    for summary in sorted(summaries, key=lambda item: item.battery_capacity_kwh):
        usable_discharge_energy_kwh = (
            summary.battery_capacity_kwh
            * (assumptions.max_soc_fraction - assumptions.min_soc_fraction)
            * discharge_efficiency
        )
        days_of_autonomy = (
            usable_discharge_energy_kwh / average_daily_load_kwh
            if average_daily_load_kwh > 0
            else 0.0
        )
        nominal_usable_autonomy_days = (
            summary.battery_capacity_kwh
            * (assumptions.max_soc_fraction - assumptions.min_soc_fraction)
            / average_daily_load_kwh
            if average_daily_load_kwh > 0
            else 0.0
        )
        nameplate_autonomy_days = (
            summary.battery_capacity_kwh / average_daily_load_kwh
            if average_daily_load_kwh > 0
            else 0.0
        )
        plot_rows.append(
            {
                "battery_capacity_kwh": summary.battery_capacity_kwh,
                "battery_power_kw": summary.battery_power_kw,
                "usable_discharge_energy_kwh": usable_discharge_energy_kwh,
                "days_of_autonomy": days_of_autonomy,
                "hours_of_autonomy": days_of_autonomy * 24.0,
                "nominal_usable_autonomy_days": nominal_usable_autonomy_days,
                "nameplate_autonomy_days": nameplate_autonomy_days,
                "annual_energy_bill_rs": summary.energy_bill_rs,
                "annualized_battery_cost_rs": summary.annualized_battery_cost_rs,
                "annualized_fixed_pv_cost_rs": annualized_fixed_pv_cost_rs,
                "annualized_total_project_cost_rs": summary.annualized_total_project_cost_rs,
                "annual_savings_vs_no_battery_rs": summary.annual_savings_vs_no_battery_rs,
            }
        )

    data_path = output_dir / "bess_autonomy_days_comparison.csv"
    with data_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    optimum = min(plot_rows, key=lambda item: item["annualized_total_project_cost_rs"])
    nonzero_rows = [row for row in plot_rows if row["battery_capacity_kwh"] > 0]
    best_nonzero = min(nonzero_rows, key=lambda item: item["annualized_total_project_cost_rs"])

    autonomy_days = [row["days_of_autonomy"] for row in plot_rows]
    total_cost_million = [row["annualized_total_project_cost_rs"] / 1_000_000 for row in plot_rows]
    bill_million = [row["annual_energy_bill_rs"] / 1_000_000 for row in plot_rows]
    battery_cost_million = [row["annualized_battery_cost_rs"] / 1_000_000 for row in plot_rows]

    chart_total_path = output_dir / "cost_vs_days_of_autonomy.svg"
    write_line_chart_svg(
        chart_total_path,
        "Annualized Project Cost vs Days of Autonomy",
        "Days of Autonomy (usable storage from max to min SOC)",
        "Annualized total project cost (Rs million/year)",
        [("Total project cost", autonomy_days, total_cost_million, "#0b6e4f")],
        (
            optimum["days_of_autonomy"],
            optimum["annualized_total_project_cost_rs"] / 1_000_000,
            f"Optimum: {optimum['days_of_autonomy']:.2f} days\nRs {optimum['annualized_total_project_cost_rs'] / 1_000_000:.2f} million/year",
        ),
    )

    chart_components_path = output_dir / "cost_components_vs_days_of_autonomy.svg"
    write_line_chart_svg(
        chart_components_path,
        "Annual Cost Components vs Days of Autonomy",
        "Days of Autonomy (usable storage from max to min SOC)",
        "Cost (Rs million/year)",
        [
            ("Annual grid bill", autonomy_days, bill_million, "#1f77b4"),
            ("Annualized BESS cost", autonomy_days, battery_cost_million, "#d97706"),
            ("Total project cost", autonomy_days, total_cost_million, "#0b6e4f"),
        ],
    )

    optimum["average_daily_load_kwh"] = average_daily_load_kwh
    optimum["lowest_cost_nonzero_bess_capacity_kwh"] = best_nonzero["battery_capacity_kwh"]
    optimum["lowest_cost_nonzero_bess_autonomy_days"] = best_nonzero["days_of_autonomy"]
    optimum["lowest_cost_nonzero_bess_project_cost_rs"] = best_nonzero["annualized_total_project_cost_rs"]
    return optimum, data_path, chart_total_path, chart_components_path


def run_trial(args: argparse.Namespace) -> None:
    assumptions = ProjectAssumptions(
        import_tariff_rs_per_kwh=args.import_tariff,
        export_credit_rs_per_kwh=args.export_credit,
    )
    pv_monthly = read_pvsyst_monthly(args.irradiance_csv)
    daily_load = read_load_profile(args.load_csv)
    hourly_rows = build_hourly_profile(pv_monthly, daily_load, args.year)

    no_battery, _ = simulate_bess(hourly_rows, 0.0, 0.0, assumptions)

    capacities = [float(value) for value in args.candidate_capacities_kwh.split(",")]
    summaries: list[DispatchSummary] = []
    fixed_case_dispatch: list[dict[str, float | int | str]] = []

    for capacity in capacities:
        power = 0.0 if capacity == 0 else min(args.battery_rating_kw, capacity / args.storage_hours)
        summary, dispatch = simulate_bess(hourly_rows, capacity, power, assumptions)
        energy_savings = no_battery.energy_bill_rs - summary.energy_bill_rs
        battery_capex = (
            capacity * assumptions.battery_cost_rs_per_kwh
            + (power / 1000.0) * assumptions.inverter_cost_rs_per_mw
        )
        summary.annual_savings_vs_no_battery_rs = energy_savings
        summary.simple_payback_years = battery_capex / energy_savings if energy_savings > 0 else None
        summaries.append(summary)

        if abs(capacity - args.fixed_battery_capacity_kwh) < 1e-9:
            fixed_case_dispatch = dispatch

    summaries.sort(key=lambda item: item.annualized_total_project_cost_rs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "bess_trial_summary.json"
    results_path = args.output_dir / "bess_capacity_comparison.csv"
    dispatch_path = args.output_dir / "bess_fixed_500kwh_dispatch.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "Synthetic hourly PV profile from monthly PVsyst E_Grid and hourly load profile from load block midpoints. Dispatch rule: PV serves load, surplus charges battery, remaining surplus exports, battery discharges to serve remaining load, grid imports final deficit.",
                "assumptions": asdict(assumptions),
                "source_totals": {
                    "annual_pv_available_kwh": sum(float(row["pv_available_kwh"]) for row in hourly_rows),
                    "annual_load_kwh": sum(float(row["load_kwh"]) for row in hourly_rows),
                    "daily_load_kwh": sum(daily_load),
                    "pvsyst_annual_e_grid_kwh": sum(pv_monthly[month]["E_Grid"] for month in MONTHS),
                },
                "no_battery_case": asdict(no_battery),
                "best_case": asdict(summaries[0]),
                "fixed_500kwh_case": asdict(next(item for item in summaries if item.battery_capacity_kwh == args.fixed_battery_capacity_kwh)),
            },
            file,
            indent=2,
        )

    with results_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(asdict(summaries[0]).keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))

    days_in_model = len(hourly_rows) / 24.0
    total_load_kwh = sum(float(row["load_kwh"]) for row in hourly_rows)

    optimum, recharge_data_path, chart_total_path, chart_components_path = create_recharge_days_outputs(
        summaries=summaries,
        no_battery=no_battery,
        assumptions=assumptions,
        days_in_model=days_in_model,
        output_dir=args.output_dir,
    )

    autonomy_optimum, autonomy_data_path, chart_autonomy_total_path, chart_autonomy_components_path = create_autonomy_days_outputs(
        summaries=summaries,
        total_load_kwh=total_load_kwh,
        assumptions=assumptions,
        days_in_model=days_in_model,
        output_dir=args.output_dir,
    )

    if fixed_case_dispatch:
        with dispatch_path.open("w", encoding="utf-8", newline="") as file:
            fieldnames = list(fixed_case_dispatch[0].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fixed_case_dispatch)

    print(f"Wrote {summary_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {dispatch_path}")
    print(f"Wrote {recharge_data_path}")
    print(f"Wrote {chart_total_path}")
    print(f"Wrote {chart_components_path}")
    print(f"Wrote {autonomy_data_path}")
    print(f"Wrote {chart_autonomy_total_path}")
    print(f"Wrote {chart_autonomy_components_path}")
    print("")
    print("Best capacity by annualized total project cost:")
    print(json.dumps(asdict(summaries[0]), indent=2))
    print("")
    print("Fixed 1 MW / 0.5 MWh case:")
    print(json.dumps(asdict(next(item for item in summaries if item.battery_capacity_kwh == args.fixed_battery_capacity_kwh)), indent=2))
    print("")
    print("Optimum point on cost vs recharge-days chart:")
    print(
        f"Recharge days = {optimum['equivalent_recharge_days']:.3f}, "
        f"annualized total project cost = Rs {optimum['annualized_total_project_cost_rs']:,.0f}/year, "
        f"battery capacity = {optimum['battery_capacity_kwh']:,.0f} kWh"
    )
    print(
        "Lowest-cost non-zero BESS point: "
        f"{optimum['lowest_cost_nonzero_bess_recharge_days']:.3f} recharge days, "
        f"Rs {optimum['lowest_cost_nonzero_bess_project_cost_rs']:,.0f}/year, "
        f"{optimum['lowest_cost_nonzero_bess_capacity_kwh']:,.0f} kWh"
    )
    print("")
    print("Optimum point on cost vs days-of-autonomy chart:")
    print(
        f"Days of autonomy = {autonomy_optimum['days_of_autonomy']:.3f} days ({autonomy_optimum['days_of_autonomy']*24:.1f} hours), "
        f"annualized total project cost = Rs {autonomy_optimum['annualized_total_project_cost_rs']:,.0f}/year, "
        f"battery capacity = {autonomy_optimum['battery_capacity_kwh']:,.0f} kWh"
    )
    print(
        "Lowest-cost non-zero BESS point (autonomy): "
        f"{autonomy_optimum['lowest_cost_nonzero_bess_autonomy_days']:.3f} days of autonomy ({autonomy_optimum['lowest_cost_nonzero_bess_autonomy_days']*24:.1f} hours), "
        f"Rs {autonomy_optimum['lowest_cost_nonzero_bess_project_cost_rs']:,.0f}/year, "
        f"{autonomy_optimum['lowest_cost_nonzero_bess_capacity_kwh']:,.0f} kWh"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initial PV+BESS optimization trial using monthly PVsyst and load-block CSVs.")
    parser.add_argument("--irradiance-csv", type=Path, default=Path(r"C:\Users\Vinayaka\OneDrive\Desktop\irradiance.csv"))
    parser.add_argument("--load-csv", type=Path, default=Path(r"C:\Users\Vinayaka\OneDrive\Desktop\load profile.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\Vinayaka\Documents\Codex\2026-09-13\i-x20\outputs"))
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--import-tariff", type=float, default=8.0)
    parser.add_argument("--export-credit", type=float, default=0.0)
    parser.add_argument("--battery-rating-kw", type=float, default=1000.0)
    parser.add_argument("--fixed-battery-capacity-kwh", type=float, default=500.0)
    parser.add_argument("--storage-hours", type=float, default=2.0)
    parser.add_argument("--candidate-capacities-kwh", default="0,500,1000,1500,2000,2500,3000,4000")
    args = parser.parse_args()
    run_trial(args)


if __name__ == "__main__":
    main()
