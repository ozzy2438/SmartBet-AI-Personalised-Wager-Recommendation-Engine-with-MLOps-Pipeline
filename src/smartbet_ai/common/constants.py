"""Constants shared across model training, inference, and monitoring."""

USER_FEATURE_COLUMNS = [
    "sport_encoded",
    "loyalty_encoded",
    "avg_stake_scaled",
    "bet_freq_scaled",
    "account_age_scaled",
    "is_high_roller",
    "total_bets",
    "win_rate",
    "sport_diversity",
]

MARKET_FEATURE_COLUMNS = [
    "sport_encoded",
    "market_type_encoded",
    "event_prominence_encoded",
    "is_live_int",
    "odds_home_scaled",
    "odds_away_scaled",
    "popularity_score",
    "implied_margin",
    "bet_count",
]

LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]
EVENT_PROMINENCE = ["low", "medium", "high", "marquee"]
OUTCOMES = ["win", "loss", "void", "cash_out"]
