"""
Generates synthetic but statistically realistic sports betting data.
Distributions are grounded in published industry heuristics and are
intended to create a useful local development dataset.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from smartbet_ai.common.config import load_data_schema, load_model_config
from smartbet_ai.common.constants import LOYALTY_TIERS, OUTCOMES
from smartbet_ai.common.paths import RAW_DATA_DIR, ensure_project_dirs

np.random.seed(42)


def load_configs() -> tuple[dict, dict]:
    """Load model and schema configs."""
    return load_model_config(), load_data_schema()


def generate_users(config: dict) -> pd.DataFrame:
    """
    Generate synthetic user profiles with realistic skew and simple feature
    correlations that the model can learn.
    """
    n = config["data"]["n_users"]
    sports = config["data"]["sports"]

    user_ids = list(range(n))
    preferred_sport = np.random.choice(
        sports,
        n,
        p=[0.20, 0.12, 0.15, 0.10, 0.08, 0.08, 0.12, 0.06, 0.05, 0.04],
    )

    avg_stake = np.random.lognormal(mean=2.8, sigma=1.2, size=n).round(2)
    avg_stake = np.clip(avg_stake, 1.0, 10000.0)

    bet_frequency = np.random.poisson(lam=4, size=n)
    bet_frequency = np.clip(bet_frequency, 0, 100)

    log_stakes = np.log(avg_stake)
    tier_probs = np.zeros((n, len(LOYALTY_TIERS)))
    for i in range(n):
        if log_stakes[i] < 2.0:
            tier_probs[i] = [0.70, 0.20, 0.08, 0.02]
        elif log_stakes[i] < 3.5:
            tier_probs[i] = [0.30, 0.40, 0.20, 0.10]
        elif log_stakes[i] < 5.0:
            tier_probs[i] = [0.10, 0.25, 0.40, 0.25]
        else:
            tier_probs[i] = [0.05, 0.10, 0.30, 0.55]

    loyalty_tier = np.array(
        [LOYALTY_TIERS[np.random.choice(len(LOYALTY_TIERS), p=probs)] for probs in tier_probs]
    )

    account_age_days = np.random.exponential(scale=365, size=n).astype(int)
    account_age_days = np.clip(account_age_days, 1, 3650)

    responsible_gambling_flag = np.random.choice([True, False], n, p=[0.03, 0.97])

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "preferred_sport": preferred_sport,
            "avg_stake": avg_stake,
            "bet_frequency_per_week": bet_frequency,
            "loyalty_tier": loyalty_tier,
            "account_age_days": account_age_days,
            "responsible_gambling_flag": responsible_gambling_flag,
        }
    )


def generate_markets(config: dict) -> pd.DataFrame:
    """Generate a synthetic market catalogue."""
    n = config["data"]["n_markets"]
    sports = config["data"]["sports"]
    market_types = config["data"]["market_types"]

    market_ids = list(range(n))
    sport = np.random.choice(
        sports,
        n,
        p=[0.20, 0.12, 0.15, 0.10, 0.08, 0.08, 0.12, 0.06, 0.05, 0.04],
    )
    market_type = np.random.choice(
        market_types,
        n,
        p=[0.35, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05],
    )
    odds_home = np.random.uniform(1.2, 8.0, n).round(2)
    odds_away = np.random.uniform(1.2, 8.0, n).round(2)
    popularity_score = np.random.beta(a=2, b=5, size=n).round(4)
    is_live = np.random.choice([True, False], n, p=[0.25, 0.75])
    event_prominence = np.random.choice(
        ["low", "medium", "high", "marquee"],
        n,
        p=[0.50, 0.30, 0.15, 0.05],
    )

    return pd.DataFrame(
        {
            "market_id": market_ids,
            "sport": sport,
            "market_type": market_type,
            "odds_home": odds_home,
            "odds_away": odds_away,
            "popularity_score": popularity_score,
            "is_live": is_live,
            "event_prominence": event_prominence,
        }
    )


def generate_interactions(users: pd.DataFrame, markets: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate non-random user-market interactions with sport preference bias,
    stake variation around user profiles, and simple temporal effects.
    """
    n_interactions = config["data"]["n_interactions"]
    weights = users["bet_frequency_per_week"].to_numpy() + 1
    weights = weights / weights.sum()

    interaction_user_ids = np.random.choice(users["user_id"].to_numpy(), n_interactions, p=weights)
    users_indexed = users.set_index("user_id")
    markets_by_sport = {sport: frame.reset_index(drop=True) for sport, frame in markets.groupby("sport")}
    now = datetime.utcnow()

    records = []
    for interaction_id, uid in enumerate(interaction_user_ids):
        user_row = users_indexed.loc[int(uid)]
        sport = user_row["preferred_sport"] if np.random.random() < 0.60 else np.random.choice(config["data"]["sports"])

        matching_markets = markets_by_sport.get(sport)
        if matching_markets is None or matching_markets.empty:
            matching_markets = markets

        market_weights = matching_markets["popularity_score"].to_numpy() + 0.01
        market_weights = market_weights / market_weights.sum()
        market_row = matching_markets.iloc[np.random.choice(len(matching_markets), p=market_weights)]

        stake = float(np.round(np.random.lognormal(mean=np.log(user_row["avg_stake"]), sigma=0.5), 2))
        stake = max(0.50, stake)

        odds = float(np.random.choice([market_row["odds_home"], market_row["odds_away"]]))
        implied_prob = 1.0 / odds
        win_prob = min(implied_prob * 0.9, 0.90)
        loss_prob = max(1.0 - win_prob - 0.02 - 0.03, 0.0)
        remainder = 1.0 - (win_prob + loss_prob + 0.02 + 0.03)
        loss_prob += remainder

        outcome = np.random.choice(OUTCOMES, p=[win_prob, loss_prob, 0.02, 0.03])

        days_ago = min(np.random.exponential(scale=30), 90)
        hour_probs = np.array(
            [
                0.02,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.05,
                0.06,
                0.06,
                0.05,
                0.05,
                0.06,
                0.06,
                0.05,
                0.06,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
            ],
            dtype=float,
        )
        hour_probs = hour_probs / hour_probs.sum()
        hour = int(np.random.choice(range(24), p=hour_probs))
        timestamp = now - timedelta(days=float(days_ago), hours=hour)

        records.append(
            {
                "interaction_id": interaction_id,
                "user_id": int(uid),
                "market_id": int(market_row["market_id"]),
                "sport": sport,
                "market_type": market_row["market_type"],
                "stake": stake,
                "odds": odds,
                "outcome": outcome,
                "timestamp": timestamp,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Generate raw user, market, and interaction data and save it to disk."""
    ensure_project_dirs()
    config, _schema = load_configs()

    print("Generating users...")
    users = generate_users(config)

    print("Generating markets...")
    markets = generate_markets(config)

    print("Generating interactions...")
    interactions = generate_interactions(users, markets, config)

    users.to_csv(RAW_DATA_DIR / "users.csv", index=False)
    markets.to_csv(RAW_DATA_DIR / "markets.csv", index=False)
    interactions.to_csv(RAW_DATA_DIR / "interactions.csv", index=False)

    print(
        f"Generated {len(users)} users, {len(markets)} markets, "
        f"{len(interactions)} interactions"
    )
    print(f"Win rate: {(interactions['outcome'] == 'win').mean():.1%}")
    print(f"Average stake: {interactions['stake'].mean():.2f} GBP")


if __name__ == "__main__":
    main()
