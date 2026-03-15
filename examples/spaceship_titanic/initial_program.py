"""Spaceship Titanic classification — predict which passengers were transported.

Given training data with passenger records and a target 'Transported' column,
build a classifier and predict on the test set.

Data fields:
  - PassengerId (gggg_pp format), HomePlanet, CryoSleep, Cabin (deck/num/side),
    Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Name
  - Transported (target, boolean)
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


# EVOLVE-BLOCK-START
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for Spaceship Titanic dataset."""
    df = df.copy()

    # Parse Cabin into deck, num, side
    cabin_split = df["Cabin"].str.split("/", expand=True)
    if cabin_split.shape[1] == 3:
        df["Deck"] = cabin_split[0]
        df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        df["Side"] = cabin_split[2]
    else:
        df["Deck"] = np.nan
        df["CabinNum"] = np.nan
        df["Side"] = np.nan

    # Parse PassengerId to get group
    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)

    # Total spending
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)

    # Drop columns we can't use directly
    df = df.drop(columns=["Cabin", "Name", "PassengerId"], errors="ignore")

    # Encode categoricals
    cat_cols = ["HomePlanet", "Destination", "Deck", "Side"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Convert booleans
    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def build_model(X_train, y_train):
    """Build and return a trained classifier."""
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
# EVOLVE-BLOCK-END


def run(train_path: str, test_path: str) -> pd.DataFrame:
    """Train model and predict on test set.

    Args:
        train_path: Path to training CSV (with Transported column).
        test_path: Path to test CSV (without Transported column).

    Returns:
        DataFrame with PassengerId and Transported columns.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_ids = test_df["PassengerId"].copy()

    y_train = train_df["Transported"].astype(int)
    train_df = train_df.drop(columns=["Transported"])

    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df)

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)

    model = build_model(X_train, y_train)
    preds = model.predict(X_test)

    return pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": preds.astype(bool),
    })


if __name__ == "__main__":
    result = run(TRAIN_PATH, TEST_PATH)
    print(f"Predictions: {len(result)} rows")
    print(result.head())
