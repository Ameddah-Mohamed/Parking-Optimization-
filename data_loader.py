"""
data_loader.py
--------------
Reads data.csv and extracts statistical distributions used by
instance_gen.py to produce realistic synthetic parking instances.
"""

import pandas as pd
import numpy as np


# ── column name mapping ────────────────────────────────────────────────────────
# Maps raw CSV column names to clean internal names
COL_MAP = {
    "Vehicle_Type":        "vehicle_type",
    "Electric_Vehicle":    "is_ev",
    "Parking_Duration":    "duration_min",
    "Spot_Size":           "spot_size",
    "Proximity_To_Exit":   "proximity_to_exit",
    "Occupancy_Rate":      "occupancy_rate",
    "Nearby_Traffic_Level":"traffic_level",
    "Entry_Time":          "entry_hour",
    "Payment_Amount":      "payment_amount",
}


def load(csv_path: str = "data.csv") -> pd.DataFrame:
    """Load and clean the raw CSV. Returns a tidy DataFrame."""
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])

    # Keep only columns we care about and rename them
    keep = [c for c in COL_MAP if c in df.columns]
    df = df[keep].rename(columns=COL_MAP)

    # Normalise vehicle_type strings
    df["vehicle_type"] = df["vehicle_type"].str.strip().str.lower()

    # entry_hour: the CSV stores it as an integer hour (0-23)
    df["entry_hour"] = pd.to_numeric(df["entry_hour"], errors="coerce")

    # duration: already in hours in the CSV
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce") * 60  # → minutes

    df = df.dropna(subset=["vehicle_type", "duration_min"])
    return df


def extract_distributions(df: pd.DataFrame) -> dict:
    """
    Derive the key probability distributions from the dataset.
    Returns a dict consumed by instance_gen.py.
    """

    # 1. Vehicle type mix  ────────────────────────────────────────────────────
    type_counts = df["vehicle_type"].value_counts(normalize=True)
    # Collapse to the 4 categories our model uses
    type_map = {
        "car":              "car",
        "motorcycle":       "motorcycle",
        "electric vehicle": "ev",
        "truck":            "car",   # treat trucks as large cars
    }
    mapped = df["vehicle_type"].map(type_map).fillna("car")
    vehicle_type_probs = mapped.value_counts(normalize=True).to_dict()

    # 2. EV fraction (from Electric_Vehicle column) ───────────────────────────
    ev_fraction = float(df["is_ev"].mean()) if "is_ev" in df.columns else 0.15

    # 3. Parking duration distribution  ──────────────────────────────────────
    durations = df["duration_min"].dropna()
    # Fit log-normal: ln(duration) ~ Normal(mu, sigma)
    log_dur = np.log(durations.clip(lower=1))
    duration_lognormal = {
        "mu":    float(log_dur.mean()),
        "sigma": float(log_dur.std()),
    }

    # 4. Arrival rate by hour  ────────────────────────────────────────────────
    if "entry_hour" in df.columns:
        hourly = df["entry_hour"].dropna().astype(int)
        arrival_by_hour = hourly.value_counts(normalize=True).sort_index().to_dict()
        # Fill any missing hours with 0
        arrival_by_hour = {h: arrival_by_hour.get(h, 0.0) for h in range(24)}
    else:
        # Uniform fallback
        arrival_by_hour = {h: 1 / 24 for h in range(24)}

    # 5. Mean inter-arrival gap (minutes)  ───────────────────────────────────
    # Rough estimate: if dataset spans ~1 year and has N rows
    n = len(df)
    span_minutes = 365 * 24 * 60
    mean_interarrival_min = span_minutes / max(n, 1)

    return {
        "vehicle_type_probs":   vehicle_type_probs,
        "ev_fraction":          ev_fraction,
        "duration_lognormal":   duration_lognormal,
        "arrival_by_hour":      arrival_by_hour,
        "mean_interarrival_min": mean_interarrival_min,
    }


def summarise(csv_path: str = "data.csv") -> None:
    """Print a quick summary — useful for sanity-checking the data."""
    df = load(csv_path)
    dist = extract_distributions(df)

    print(f"\n{'='*50}")
    print(f"  Dataset: {len(df)} records loaded from {csv_path}")
    print(f"{'='*50}")
    print("\nVehicle type mix:")
    for k, v in sorted(dist["vehicle_type_probs"].items(), key=lambda x: -x[1]):
        print(f"  {k:<15} {v*100:.1f}%")
    print(f"\nEV fraction:       {dist['ev_fraction']*100:.1f}%")
    ln = dist["duration_lognormal"]
    print(f"Duration (log-normal):  mu={ln['mu']:.2f}, sigma={ln['sigma']:.2f}")
    median_dur = np.exp(ln["mu"])
    print(f"  → median duration ~{median_dur:.0f} min ({median_dur/60:.1f} h)")
    print(f"\nMean inter-arrival: {dist['mean_interarrival_min']:.1f} min")
    print()


if __name__ == "__main__":
    summarise()
