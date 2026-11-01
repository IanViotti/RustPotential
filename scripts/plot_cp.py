import os
import sys
import math
import glob
import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_latest_csv(base_dir="job_files"):
    """Finds the most recently modified Cp_distribution.csv inside job_files/."""
    pattern = os.path.join(base_dir, "**", "Cp_distribution.csv")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_run_config(csv_path):
    """
    Loads flight and job configuration (jobname, Mach_inf, AoA, gamma) from
    job_config.toml located in the same run folder (or root job_config.toml fallback).
    """
    run_dir = os.path.dirname(os.path.abspath(csv_path))
    candidates = [
        os.path.join(run_dir, "job_config.toml"),
        "job_config.toml",
    ]

    config = {
        "jobname": os.path.basename(run_dir) or "Airfoil",
        "Mach_inf": 0.0,
        "AoA": 0.0,
        "gamma": 1.4,
    }

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                config["jobname"] = data.get("jobname", config["jobname"])
                config["Mach_inf"] = float(data.get("Mach_inf", config["Mach_inf"]))
                aoa = data.get("AoA", data.get("alpha_deg", data.get("alpha", data.get("aoa", 0.0))))
                config["AoA"] = float(aoa)
                config["gamma"] = float(data.get("gamma", config["gamma"]))
                break
            except Exception as e:
                print(f"Warning: Could not parse {path}: {e}")

    return config


def compute_aerodynamic_coefficients(df, aoa_deg=0.0, mach_inf=0.0, gamma=1.4):
    """
    Integrates the surface pressure coefficient (Cp) around the closed airfoil contour
    using the trapezoidal rule to obtain Cn, Ca, Cl, Cd, and Cm_{c/4}.

    Contour convention in Cp_distribution.csv:
      - Counter-clockwise O-grid loop starting at TE (x ~ 1), moving along the
        upper surface to LE (x ~ 0), and returning along the lower surface to TE.
    """
    x = df["x"].to_numpy(dtype=float).copy()
    y = df["y"].to_numpy(dtype=float).copy()
    cp = df["Cp"].to_numpy(dtype=float).copy()

    # Guard against legacy trailing-edge cusp singularity at index 0 (x ~ 1.0)
    if len(cp) > 3 and abs(x[0] - 1.0) < 1e-4:
        neighbor_avg = 0.5 * (cp[1] + cp[-1])
        if abs(cp[0] - neighbor_avg) > 0.5:
            cp[0] = neighbor_avg

    # Close the contour loop back to the starting trailing-edge node
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    cp_closed = np.append(cp, cp[0])

    chord = float(np.max(x) - np.min(x))
    if chord <= 1e-12:
        chord = 1.0
    x_le = float(np.min(x))
    x_ref = x_le + 0.25 * chord  # Quarter-chord reference point
    y_ref = 0.0

    dx = np.diff(x_closed)
    dy = np.diff(y_closed)
    cp_mid = 0.5 * (cp_closed[:-1] + cp_closed[1:])
    x_mid = 0.5 * (x_closed[:-1] + x_closed[1:])
    y_mid = 0.5 * (y_closed[:-1] + y_closed[1:])

    # Force coefficients in airfoil body axes (normalized by chord)
    cn = float(np.sum(cp_mid * dx) / chord)
    ca = float(-np.sum(cp_mid * dy) / chord)

    # Project into wind axes (Lift Cl and Drag Cd)
    alpha_rad = math.radians(aoa_deg)
    cl = cn * math.cos(alpha_rad) - ca * math.sin(alpha_rad)
    cd = cn * math.sin(alpha_rad) + ca * math.cos(alpha_rad)

    # Pitching moment coefficient about quarter-chord (nose-up positive)
    cm_c4 = float(
        np.sum(cp_mid * (x_ref - x_mid) * dx + cp_mid * (y_mid - y_ref) * dy)
        / (chord ** 2)
    )

    # Extrema
    idx_min = int(np.argmin(cp))
    idx_max = int(np.argmax(cp))
    cp_min = float(cp[idx_min])
    x_cp_min = float((x[idx_min] - x_le) / chord)
    cp_max = float(cp[idx_max])
    x_cp_max = float((x[idx_max] - x_le) / chord)

    # Critical pressure coefficient Cp* (sonic condition M = 1)
    cp_star = None
    if mach_inf > 0.0:
        ratio = (2.0 + (gamma - 1.0) * (mach_inf ** 2)) / (gamma + 1.0)
        cp_star = (2.0 / (gamma * (mach_inf ** 2))) * (
            (ratio ** (gamma / (gamma - 1.0))) - 1.0
        )

    return {
        "Cl": cl,
        "Cd": cd,
        "Cn": cn,
        "Ca": ca,
        "Cm_c4": cm_c4,
        "Cp_min": cp_min,
        "x_Cp_min": x_cp_min,
        "Cp_max": cp_max,
        "x_Cp_max": x_cp_max,
        "Cp_star": cp_star,
        "cp_clean": cp,
    }


def plot_cp_distribution(csv_filename, show_plot=True):
    if os.path.isdir(csv_filename):
        csv_filename = os.path.join(csv_filename, "Cp_distribution.csv")

    print(f"Reading surface pressure data from: {csv_filename}...")

    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    config = load_run_config(csv_filename)
    aero = compute_aerodynamic_coefficients(
        df,
        aoa_deg=config["AoA"],
        mach_inf=config["Mach_inf"],
        gamma=config["gamma"],
    )
    df = df.copy()
    df["Cp"] = aero["cp_clean"]

    # Print aerodynamic summary to console
    print("-" * 65)
    print(f"  AERODYNAMIC COEFFICIENTS SUMMARY ({config['jobname']})")
    print("-" * 65)
    print(f"  Freestream Mach (M_inf) : {config['Mach_inf']:.4f}")
    print(f"  Angle of Attack (AoA)   : {config['AoA']:.2f} deg")
    print(f"  Lift Coefficient (Cl)   : {aero['Cl']:+.4f}")
    print(f"  Drag Coefficient (Cd)   : {aero['Cd']:+.4f}")
    print(f"  Moment Coeff (Cm_c/4)   : {aero['Cm_c4']:+.4f}")
    print(f"  Normal Force (Cn)       : {aero['Cn']:+.4f}")
    print(f"  Axial Force (Ca)        : {aero['Ca']:+.4f}")
    print(f"  Suction Peak (Cp_min)   : {aero['Cp_min']:+.4f} (at x/c = {aero['x_Cp_min']:.3f})")
    print(f"  Stagnation (Cp_max)     : {aero['Cp_max']:+.4f} (at x/c = {aero['x_Cp_max']:.3f})")
    if aero["Cp_star"] is not None:
        print(f"  Critical Sonic (Cp*)    : {aero['Cp_star']:+.4f}")
    print("-" * 65)

    # Split into Upper Surface and Lower Surface at the leading edge (minimum x)
    idx_le = int(df["x"].idxmin())
    upper = df.iloc[: idx_le + 1].sort_values(by="x")
    lower = pd.concat([df.iloc[idx_le:], df.iloc[[0]]]).sort_values(by="x")

    # Configure figure
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Plot Upper and Lower surface Cp curves
    ax.plot(
        upper["x"],
        upper["Cp"],
        label="Upper Surface (Suction)",
        color="#1f77b4",
        marker="o",
        markersize=3.5,
        linestyle="-",
        linewidth=1.6,
    )

    ax.plot(
        lower["x"],
        lower["Cp"],
        label="Lower Surface (Pressure)",
        color="#d62728",
        marker="s",
        markersize=3.5,
        linestyle="-",
        linewidth=1.6,
    )

    # Plot critical Cp* line if within or near the plotted Cp range
    if aero["Cp_star"] is not None and aero["Cp_min"] <= aero["Cp_star"] + 0.25:
        ax.axhline(
            y=aero["Cp_star"],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.2,
            label=f"Sonic $C_p^*$ = {aero['Cp_star']:+.3f} ($M = 1$)",
        )

    # Invert Y-axis (standard aerodynamic convention: negative Cp upward)
    ax.invert_yaxis()

    # Titles and labels
    ax.set_title(
        f"Surface Pressure Coefficient ($C_p$) — {config['jobname']} "
        f"($M_\\infty = {config['Mach_inf']:.3f}$, $\\mathrm{{AoA}} = {config['AoA']:.2f}^\\circ$)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Chordwise Position ($x/c$)", fontsize=12)
    ax.set_ylabel("Pressure Coefficient ($C_p$)", fontsize=12)

    # Embed aerodynamic coefficients box inside the plot
    stats_lines = [
        "Aerodynamic Coefficients",
        "────────────────────────",
        f"$M_\\infty$ = {config['Mach_inf']:.3f},  AoA = {config['AoA']:.2f}°",
        f"$C_l$     = {aero['Cl']:+.4f}",
        f"$C_d$     = {aero['Cd']:+.4f}",
        f"$C_{{m,c/4}}$ = {aero['Cm_c4']:+.4f}",
        f"$C_n$     = {aero['Cn']:+.4f}",
        f"$C_a$     = {aero['Ca']:+.4f}",
        f"$C_{{p,\\min}}$ = {aero['Cp_min']:+.4f} ($x/c = {aero['x_Cp_min']:.2f}$)",
        f"$C_{{p,\\max}}$ = {aero['Cp_max']:+.4f} ($x/c = {aero['x_Cp_max']:.2f}$)",
    ]
    if aero["Cp_star"] is not None:
        stats_lines.append(f"$C_p^*$     = {aero['Cp_star']:+.4f}")

    stats_text = "\n".join(stats_lines)
    bbox_props = dict(
        boxstyle="round,pad=0.55",
        facecolor="#f8f9fa",
        edgecolor="#495057",
        alpha=0.92,
        linewidth=1.0,
    )
    ax.text(
        0.98,
        0.84,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
        family="monospace",
    )

    # Grid and legend
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.92)

    fig.tight_layout()

    # Save figure in the same directory as the CSV
    output_img = os.path.splitext(csv_filename)[0] + ".png"
    fig.savefig(output_img, dpi=300)
    print(f"Plot with aerodynamic coefficients saved to: {output_img}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Usage:
    #   python scripts/plot_cp.py [optional_path_to_csv_or_job_folder] [--no-show]
    args = [a for a in sys.argv[1:] if a != "--no-show"]
    show = "--no-show" not in sys.argv[1:]

    if args:
        target_csv = args[0]
    else:
        target_csv = find_latest_csv("job_files")
        if target_csv is None:
            target_csv = "job_files/naca0012/Cp_distribution.csv"

    plot_cp_distribution(target_csv, show_plot=show)
