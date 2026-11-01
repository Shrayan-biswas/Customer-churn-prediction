from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "customer_data.csv"

def main():
    rng = np.random.default_rng(42)
    n = 1200

    tenure = rng.integers(1, 73, n)
    monthly_charges = rng.uniform(20, 120, n).round(2)
    support_calls = rng.integers(0, 10, n)
    contract_type = rng.choice(
        ["Monthly", "Yearly", "Two-Year"], n, p=[0.55, 0.30, 0.15]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber", "No"], n, p=[0.35, 0.50, 0.15]
    )

    logit = (
        -0.6
        + 0.24 * support_calls
        - 0.035 * tenure
        + 1.0 * (contract_type == "Monthly")
        - 0.45 * (contract_type == "Two-Year")
        + 0.012 * (monthly_charges - 70)
        + 0.25 * (internet_service == "Fiber")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "churn": churn,
    })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} customer records: {DATA_PATH}")

if __name__ == "__main__":
    main()
