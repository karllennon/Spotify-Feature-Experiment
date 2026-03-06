import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ── EXPERIMENT PARAMETERS ──────────────────────────────────────────────────

N_USERS = 10_000
CONTROL_SPLIT = 0.50
TREATMENT_SPLIT = 0.50

# Stage 1: Feature Adoption
CONTROL_ADOPTION_RATE = 0.05
TREATMENT_ADOPTION_RATE = 0.25

# Stage 2: Notification Re-engagement
CONTROL_REENGAGEMENT_RATE = 0.08
TREATMENT_REENGAGEMENT_RATE = 0.20

# Stage 3: Premium Conversion
CONTROL_CONVERSION_RATE = 0.10
TREATMENT_CONVERSION_RATE = 0.18


# Assigns users to control/treatment via random sampling with equal probability.
def generate_users(n_users: int = N_USERS) -> pd.DataFrame:
    user_ids = np.arange(1, n_users + 1)
    groups = np.random.choice(
        ["control", "treatment"],
        size=n_users,
        p=[CONTROL_SPLIT, TREATMENT_SPLIT]
    )

    return pd.DataFrame({
        "user_id": user_ids,
        "group": groups
    })


# Stage 1: Adoption modeled as a Bernoulli trial with group-specific rates.
def simulate_feature_adoption(users: pd.DataFrame) -> pd.DataFrame:

    adoption_rates = {
        "control": CONTROL_ADOPTION_RATE,
        "treatment": TREATMENT_ADOPTION_RATE
    }

    users["adopted"] = users["group"].apply(
        lambda group: np.random.binomial(1, adoption_rates[group])
    )

    return users


# Stage 2: Re-engagement is only modeled for non-adopters.
# Adopters are excluded as they would not receive a notification campaign.
def simulate_reengagement(users: pd.DataFrame) -> pd.DataFrame:
    reengagement_rates = {
        "control": CONTROL_REENGAGEMENT_RATE,
        "treatment": TREATMENT_REENGAGEMENT_RATE
    }

    non_adopters = users["adopted"] == 0

    users.loc[non_adopters, "reengaged"] = users.loc[non_adopters, "group"].apply(
        lambda group: np.random.binomial(1, reengagement_rates[group])
    )

    users["reengaged"] = users["reengaged"].fillna(0).astype(int)

    return users


# Stage 3: Conversion is only modeled for users who adopted the feature.
# Non-adopters are excluded as they would not receive a Premium upgrade prompt.
def simulate_conversion(users: pd.DataFrame) -> pd.DataFrame:
    conversion_rates = {
        "control": CONTROL_CONVERSION_RATE,
        "treatment": TREATMENT_CONVERSION_RATE
    }

    adopters = users["adopted"] == 1

    users.loc[adopters, "converted"] = users.loc[adopters, "group"].apply(
        lambda group: np.random.binomial(1, conversion_rates[group])
    )

    users["converted"] = users["converted"].fillna(0).astype(int)

    return users


# Orchestrates the full three-stage funnel simulation.
# Each stage filters on the output of the previous, preserving funnel integrity.
def run_simulation() -> pd.DataFrame:
    users = generate_users()
    users = simulate_feature_adoption(users)
    users = simulate_reengagement(users)
    users = simulate_conversion(users)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "simulated_users.csv")
    users.to_csv(data_path, index=False)

    return users


if __name__ == "__main__":
    df = run_simulation()
    print(df.head(10))
    print(f"\nTotal users: {len(df)}")
    print(f"Adopted: {df['adopted'].sum()}")
    print(f"Reengaged: {df['reengaged'].sum()}")
    print(f"Converted: {df['converted'].sum()}")