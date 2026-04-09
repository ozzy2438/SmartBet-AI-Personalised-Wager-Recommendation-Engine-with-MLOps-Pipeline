"""
PyTorch Dataset for the two-tower recommender with negative sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from smartbet_ai.common.constants import MARKET_FEATURE_COLUMNS, USER_FEATURE_COLUMNS


class BettingRecommendationDataset(Dataset):
    """
    Returns (user_id, user_features, market_id, market_features, label) samples.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        n_negatives: int = 4,
        split: str = "train",
    ) -> None:
        self.interactions = interactions_df.reset_index(drop=True)
        self.users = users_df.set_index("user_id")
        self.markets = markets_df.set_index("market_id")
        self.n_negatives = n_negatives
        self.all_market_ids = set(markets_df["market_id"].astype(int).tolist())
        self.user_positives = (
            self.interactions.groupby("user_id")["market_id"].apply(lambda values: set(values.astype(int))).to_dict()
        )
        self.user_feature_cols = USER_FEATURE_COLUMNS
        self.market_feature_cols = MARKET_FEATURE_COLUMNS
        self.samples = self._build_samples()

        print(
            f"{split} dataset: {len(self.samples)} samples "
            f"({len(self.interactions)} positive + {len(self.samples) - len(self.interactions)} negative)"
        )

    def _build_samples(self) -> list[dict[str, float]]:
        samples: list[dict[str, float]] = []

        for _, row in self.interactions.iterrows():
            uid = int(row["user_id"])
            mid = int(row["market_id"])
            samples.append({"user_id": uid, "market_id": mid, "label": 1.0})

            user_interacted = self.user_positives.get(uid, set())
            possible_negatives = list(self.all_market_ids - user_interacted)
            if not possible_negatives:
                continue

            n_to_sample = min(self.n_negatives, len(possible_negatives))
            negative_market_ids = np.random.choice(possible_negatives, n_to_sample, replace=False)
            for negative_market_id in negative_market_ids:
                samples.append(
                    {
                        "user_id": uid,
                        "market_id": int(negative_market_id),
                        "label": 0.0,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        uid = sample["user_id"]
        mid = sample["market_id"]
        label = sample["label"]

        try:
            user_row = self.users.loc[uid]
            if isinstance(user_row, pd.DataFrame):
                user_row = user_row.iloc[0]
            user_features = torch.tensor(
                [float(user_row[column]) for column in self.user_feature_cols],
                dtype=torch.float32,
            )
        except (KeyError, IndexError):
            user_features = torch.zeros(len(self.user_feature_cols), dtype=torch.float32)

        try:
            market_row = self.markets.loc[mid]
            if isinstance(market_row, pd.DataFrame):
                market_row = market_row.iloc[0]
            market_features = torch.tensor(
                [float(market_row[column]) for column in self.market_feature_cols],
                dtype=torch.float32,
            )
        except (KeyError, IndexError):
            market_features = torch.zeros(len(self.market_feature_cols), dtype=torch.float32)

        return (
            torch.tensor(uid, dtype=torch.long),
            user_features,
            torch.tensor(mid, dtype=torch.long),
            market_features,
            torch.tensor(label, dtype=torch.float32),
        )
