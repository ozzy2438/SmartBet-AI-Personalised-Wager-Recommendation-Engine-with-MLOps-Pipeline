"""
Transforms raw data into model-ready features for the user and market towers.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from smartbet_ai.common.config import load_model_config
from smartbet_ai.common.constants import (
    EVENT_PROMINENCE,
    LOYALTY_TIERS,
    MARKET_FEATURE_COLUMNS,
    USER_FEATURE_COLUMNS,
)
from smartbet_ai.common.paths import FEATURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs


def engineer_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create processed datasets and persist feature artifacts for serving."""
    ensure_project_dirs()
    config = load_model_config()
    sports = config["data"]["sports"]
    market_types = config["data"]["market_types"]

    print("Loading raw data...")
    users = pd.read_csv(RAW_DATA_DIR / "users.csv")
    markets = pd.read_csv(RAW_DATA_DIR / "markets.csv")
    interactions = pd.read_csv(RAW_DATA_DIR / "interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    sport_encoder = LabelEncoder().fit(sports)
    market_type_encoder = LabelEncoder().fit(market_types)
    loyalty_encoder = LabelEncoder().fit(LOYALTY_TIERS)
    prominence_encoder = LabelEncoder().fit(EVENT_PROMINENCE)

    print("Engineering user features...")
    users["sport_encoded"] = sport_encoder.transform(users["preferred_sport"])
    users["loyalty_encoded"] = loyalty_encoder.transform(users["loyalty_tier"])

    stake_scaler = StandardScaler()
    users["avg_stake_scaled"] = stake_scaler.fit_transform(users[["avg_stake"]]).ravel()

    freq_scaler = StandardScaler()
    users["bet_freq_scaled"] = freq_scaler.fit_transform(users[["bet_frequency_per_week"]]).ravel()

    age_scaler = StandardScaler()
    users["account_age_scaled"] = age_scaler.fit_transform(users[["account_age_days"]]).ravel()

    high_roller_threshold = users["avg_stake"].quantile(0.90)
    users["is_high_roller"] = (users["avg_stake"] >= high_roller_threshold).astype(int)

    print("Engineering market features...")
    markets["sport_encoded"] = sport_encoder.transform(markets["sport"])
    markets["market_type_encoded"] = market_type_encoder.transform(markets["market_type"])
    markets["event_prominence_encoded"] = prominence_encoder.transform(markets["event_prominence"])
    markets["is_live_int"] = markets["is_live"].astype(int)

    odds_home_scaler = StandardScaler()
    markets["odds_home_scaled"] = odds_home_scaler.fit_transform(markets[["odds_home"]]).ravel()

    odds_away_scaler = StandardScaler()
    markets["odds_away_scaled"] = odds_away_scaler.fit_transform(markets[["odds_away"]]).ravel()

    markets["implied_margin"] = (1 / markets["odds_home"] + 1 / markets["odds_away"] - 1).round(4)

    print("Engineering interaction features...")
    interactions["sport_encoded"] = sport_encoder.transform(interactions["sport"])
    interactions["market_type_encoded"] = market_type_encoder.transform(interactions["market_type"])
    interactions["day_of_week"] = interactions["timestamp"].dt.dayofweek
    interactions["hour_of_day"] = interactions["timestamp"].dt.hour
    interactions["is_weekend"] = (interactions["day_of_week"] >= 5).astype(int)
    interactions["label"] = 1.0

    user_history = interactions.groupby("user_id").agg(
        total_bets=("interaction_id", "count"),
        total_staked=("stake", "sum"),
        win_rate=("outcome", lambda values: (values == "win").mean()),
        avg_odds=("odds", "mean"),
        sport_diversity=("sport", "nunique"),
        market_type_diversity=("market_type", "nunique"),
    )

    users = users.merge(user_history.reset_index(), on="user_id", how="left")
    for column in [
        "total_bets",
        "total_staked",
        "win_rate",
        "avg_odds",
        "sport_diversity",
        "market_type_diversity",
    ]:
        users[column] = users[column].fillna(0)

    market_popularity = interactions.groupby("market_id").agg(
        bet_count=("interaction_id", "count"),
        total_volume=("stake", "sum"),
        avg_bet_size=("stake", "mean"),
    )
    markets = markets.merge(market_popularity.reset_index(), on="market_id", how="left")
    for column in ["bet_count", "total_volume", "avg_bet_size"]:
        markets[column] = markets[column].fillna(0)

    joblib.dump(sport_encoder, FEATURES_DIR / "sport_encoder.pkl")
    joblib.dump(market_type_encoder, FEATURES_DIR / "market_type_encoder.pkl")
    joblib.dump(loyalty_encoder, FEATURES_DIR / "loyalty_encoder.pkl")
    joblib.dump(prominence_encoder, FEATURES_DIR / "prominence_encoder.pkl")
    joblib.dump(stake_scaler, FEATURES_DIR / "stake_scaler.pkl")
    joblib.dump(freq_scaler, FEATURES_DIR / "freq_scaler.pkl")
    joblib.dump(age_scaler, FEATURES_DIR / "age_scaler.pkl")
    joblib.dump(odds_home_scaler, FEATURES_DIR / "odds_home_scaler.pkl")
    joblib.dump(odds_away_scaler, FEATURES_DIR / "odds_away_scaler.pkl")

    feature_manifest = {
        "user_feature_columns": USER_FEATURE_COLUMNS,
        "market_feature_columns": MARKET_FEATURE_COLUMNS,
    }
    with (FEATURES_DIR / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_manifest, handle, indent=2)

    users.to_csv(PROCESSED_DATA_DIR / "users_processed.csv", index=False)
    markets.to_csv(PROCESSED_DATA_DIR / "markets_processed.csv", index=False)
    interactions.to_csv(PROCESSED_DATA_DIR / "interactions_processed.csv", index=False)

    print("Feature engineering complete.")
    print(f"Users: {len(users)} rows, {users.shape[1]} columns")
    print(f"Markets: {len(markets)} rows, {markets.shape[1]} columns")
    print(f"Interactions: {len(interactions)} rows")

    return users, markets, interactions


if __name__ == "__main__":
    engineer_features()
