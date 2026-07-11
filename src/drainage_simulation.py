"""
drainage_simulation.py
------------------------
Module 2: Drainage Simulation using the Rational Method.

Rational Method formula:
    Q = C * I * A / 360      (Q in m3/s, when I in mm/hr and A in hectares)

Where:
    Q = peak runoff (m3/s)
    C = runoff coefficient (0-1, depends on land type: concrete, soil, etc.)
    I = rainfall intensity (mm/hr)
    A = catchment / drainage area (hectares)

We simulate a 48-hour rainfall event over an area with a fixed drain
network capacity, and detect the hours where runoff EXCEEDS drainage
capacity (i.e. waterlogging / flood points).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_rainfall_event(hours: int = 48, seed: int = 7) -> np.ndarray:
    """Generate a realistic rainfall intensity time series (mm/hr)."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    # base storm pattern: intensity rises then falls (like a real storm)
    storm_shape = np.exp(-((t - hours * 0.4) ** 2) / (2 * (hours * 0.12) ** 2))
    intensity = 15 * storm_shape + rng.normal(0, 1.5, hours).clip(min=-2)
    intensity = np.clip(intensity, 0, None)
    return intensity


def rational_method_runoff(intensity_mm_hr, C: float, A_hectares: float) -> np.ndarray:
    """Compute runoff Q (m3/s) using the Rational Method for each time step."""
    Q = (C * intensity_mm_hr * A_hectares) / 360
    return Q


def run_drainage_simulation(output_dir: str,
                             C: float = 0.65,
                             A_hectares: float = 120,
                             drainage_capacity_m3s: float = 3.0,
                             hours: int = 48) -> pd.DataFrame:
    """
    C: runoff coefficient (0.65 = mixed urban/semi-urban area)
    A_hectares: catchment area draining into this system
    drainage_capacity_m3s: max flow the drain/canal network can carry
    """
    intensity = simulate_rainfall_event(hours=hours)
    runoff = rational_method_runoff(intensity, C, A_hectares)
    overflow = runoff > drainage_capacity_m3s

    df = pd.DataFrame({
        "Hour": np.arange(hours),
        "RainfallIntensity_mm_hr": intensity.round(2),
        "Runoff_m3s": runoff.round(2),
        "DrainageCapacity_m3s": drainage_capacity_m3s,
        "Overflow": overflow,
    })

    # --- Plot: rainfall intensity ---
    plt.figure(figsize=(10, 4))
    plt.plot(df["Hour"], df["RainfallIntensity_mm_hr"], color="#2E86AB")
    plt.fill_between(df["Hour"], df["RainfallIntensity_mm_hr"], color="#2E86AB", alpha=0.2)
    plt.title("Simulated Rainfall Event (48 hours)")
    plt.xlabel("Hour")
    plt.ylabel("Rainfall Intensity (mm/hr)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_rainfall_event.png", dpi=150)
    plt.close()

    # --- Plot: runoff vs drainage capacity ---
    plt.figure(figsize=(10, 5))
    plt.plot(df["Hour"], df["Runoff_m3s"], label="Runoff (Rational Method)", color="#D64545")
    plt.axhline(drainage_capacity_m3s, color="black", linestyle="--",
                label=f"Drainage Capacity = {drainage_capacity_m3s} m3/s")
    plt.fill_between(df["Hour"], df["Runoff_m3s"], drainage_capacity_m3s,
                      where=(df["Runoff_m3s"] > drainage_capacity_m3s),
                      color="red", alpha=0.3, label="Overflow (waterlogging risk)")
    plt.title("Drainage Simulation: Runoff vs Drainage Capacity")
    plt.xlabel("Hour")
    plt.ylabel("Flow (m3/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/7_drainage_simulation.png", dpi=150)
    plt.close()

    overflow_hours = df[df["Overflow"]]["Hour"].tolist()
    print(f"Overflow (waterlogging) predicted at hours: {overflow_hours}")
    print(f"Total overflow duration: {len(overflow_hours)} hour(s) out of {hours}")

    return df
