import pandas as pd
import numpy as np
import os
import logging
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
TRAINING_DATA_CSV = os.path.join(BASE_DIR, "data", "vsechny_evropske_zapasy.csv")
REAL_RESULTS_CSV = os.path.join(BASE_DIR, "data", "realne_vysledky.csv")
FEATURES = [
    "PrevRatingA",
    "PrevRatingB",
    "HomeAdvantage",
    "NeutralGround",
    "IsFriendly",
    "IsQualifier",
    "IsMainTournament",
]
MAIN_TOURNAMENTS = [
    "Confederations Cup",
    "European Championship",
    "European Nations League",
    "Intercontinental Championship",
    "King's Cup",
    "World Cup",
]

HARDCODED_INITIAL_RATINGS = {
    "Albania": 1651,
    "Austria": 1805,
    "Belgium": 1939,
    "Croatia": 1850,
    "Czechia": 1735,
    "Denmark": 1840,
    "England": 2088,
    "France": 2068,
    "Georgia": 1641,
    "Germany": 2033,
    "Hungary": 1676,
    "Italy": 1949,
    "Netherlands": 1959,
    "Poland": 1825,
    "Portugal": 1994,
    "Romania": 1616,
    "Scotland": 1765,
    "Serbia": 1810,
    "Slovakia": 1681,
    "Slovenia": 1686,
    "Spain": 2028,
    "Switzerland": 1845,
    "Turkey": 1810,
    "Ukraine": 1800,
}


def train_models(training_csv_path):
    """Loads training data and trains Poisson regression models."""
    try:
        training_data = pd.read_csv(training_csv_path)
        logging.info(f"Načtena trénovací data z {training_csv_path}")
    except FileNotFoundError:
        logging.error(f"Trénovací soubor CSV nebyl nalezen na {training_csv_path}")
        return None, None
    except Exception as e:
        logging.error(f"Chyba při načítání trénovacích dat CSV: {e}")
        return None, None

    X_train = training_data.copy()
    prev_rating_a = X_train["NewRatingA"].fillna(
        X_train["NewRatingA"].mean()
    ) - X_train["RatingChangeA"].fillna(0)
    prev_rating_b = X_train["NewRatingB"].fillna(
        X_train["NewRatingB"].mean()
    ) - X_train["RatingChangeB"].fillna(0)
    X_train["PrevRatingA"] = prev_rating_a
    X_train["PrevRatingB"] = prev_rating_b

    X_train["HomeAdvantage"] = (X_train["Location"] == X_train["TeamA"]).astype(int)
    X_train["NeutralGround"] = (
        (X_train["Location"] != X_train["TeamA"])
        & (X_train["Location"] != X_train["TeamB"])
    ).astype(int)
    X_train["IsFriendly"] = (
        X_train["Tournament"].str.contains("Friendly", case=False, na=False).astype(int)
    )
    X_train["IsQualifier"] = (
        X_train["Tournament"]
        .str.contains("qualifier", case=False, na=False)
        .astype(int)
    )
    X_train["IsMainTournament"] = (
        X_train["Tournament"].isin(MAIN_TOURNAMENTS).astype(int)
    )

    y_train_a = training_data["ScoreA"]
    y_train_b = training_data["ScoreB"]

    X_train_features = X_train[FEATURES].fillna(X_train[FEATURES].mean())
    y_train_a = y_train_a.fillna(y_train_a.median())
    y_train_b = y_train_b.fillna(y_train_b.median())

    X_train_features = X_train_features.replace([np.inf, -np.inf], np.nan).fillna(
        X_train_features.mean()
    )
    y_train_a = y_train_a.replace([np.inf, -np.inf], np.nan).fillna(y_train_a.median())
    y_train_b = y_train_b.replace([np.inf, -np.inf], np.nan).fillna(y_train_b.median())

    try:
        model_a = PoissonRegressor(alpha=0.1, max_iter=1000)
        model_b = PoissonRegressor(alpha=0.1, max_iter=1000)
        model_a.fit(X_train_features, y_train_a)
        model_b.fit(X_train_features, y_train_b)
        logging.info("Modely Poissonovy regrese úspěšně natrénovány.")
        return model_a, model_b
    except Exception as e:
        logging.error(f"Chyba při trénování modelů: {e}")
        return None, None


def test_prediction_accuracy():
    logging.info("Spouštění testu přesnosti predikce...")

    initial_ratings = HARDCODED_INITIAL_RATINGS
    logging.info(
        f"Používají se pevně zakódovaná počáteční hodnocení pro {len(initial_ratings)} týmů."
    )
    logging.info(f"Trénování modelů pomocí {TRAINING_DATA_CSV}...")
    model_a, model_b = train_models(TRAINING_DATA_CSV)
    if model_a is None or model_b is None:
        logging.error("Nepodařilo se natrénovat predikční modely. Skript končí.")
        return

    try:
        real_results_df = pd.read_csv(REAL_RESULTS_CSV)
        logging.info(f"Načtena skutečná data výsledků z {REAL_RESULTS_CSV}")
        real_results_df["RealA"] = pd.to_numeric(
            real_results_df["RealA"], errors="coerce"
        )
        real_results_df["RealB"] = pd.to_numeric(
            real_results_df["RealB"], errors="coerce"
        )
        real_results_df.dropna(subset=["RealA", "RealB"], inplace=True)
        real_results_df["RealA"] = real_results_df["RealA"].astype(int)
        real_results_df["RealB"] = real_results_df["RealB"].astype(int)

    except FileNotFoundError:
        logging.error(
            f"Soubor se skutečnými výsledky nebyl nalezen na {REAL_RESULTS_CSV}"
        )
        return
    except Exception as e:
        logging.error(
            f"Chyba při načítání nebo zpracování souboru se skutečnými výsledky: {e}"
        )
        return

    predictions = []
    actuals_a = []
    actuals_b = []
    predicted_lambda_a = []
    predicted_lambda_b = []

    logging.info("Predikce výsledků pro skutečné zápasy...")
    for index, row in real_results_df.iterrows():
        team_a = row["TeamA"]
        team_b = row["TeamB"]
        location = row["Location"]
        tournament = row["Tournament"]

        rating_a = initial_ratings.get(team_a)
        rating_b = initial_ratings.get(team_b)

        if rating_a is None or rating_b is None:
            logging.warning(
                f"Přeskakuji zápas: Chybí počáteční rating pro {team_a} ({rating_a}) nebo {team_b} ({rating_b})."
            )
            continue

        is_home_a = location == team_a
        is_neutral = (location != team_a) and (location != team_b)
        is_main_tournament = tournament in MAIN_TOURNAMENTS

        match_features = {
            "PrevRatingA": [rating_a],
            "PrevRatingB": [rating_b],
            "HomeAdvantage": [int(is_home_a)],
            "NeutralGround": [int(is_neutral)],
            "IsFriendly": [0],
            "IsQualifier": [0],
            "IsMainTournament": [int(is_main_tournament)],
        }
        X_pred = pd.DataFrame(match_features)

        try:
            lambda_a = max(0.01, model_a.predict(X_pred[FEATURES])[0])
            lambda_b = max(0.01, model_b.predict(X_pred[FEATURES])[0])
        except Exception as e:
            logging.error(f"Chyba při predikci pro zápas {team_a} vs {team_b}: {e}")
            continue

        actuals_a.append(row["RealA"])
        actuals_b.append(row["RealB"])
        predicted_lambda_a.append(lambda_a)
        predicted_lambda_b.append(lambda_b)

    if not actuals_a:
        logging.error(
            "Nebyly provedeny žádné platné predikce. Zkontrolujte data a ratingy."
        )
        return

    logging.info("Výpočet metrik přesnosti...")

    mae_a = mean_absolute_error(actuals_a, predicted_lambda_a)
    mae_b = mean_absolute_error(actuals_b, predicted_lambda_b)

    predicted_score_a_rounded = np.round(predicted_lambda_a).astype(int)
    predicted_score_b_rounded = np.round(predicted_lambda_b).astype(int)
    exact_matches = sum(
        (predicted_score_a_rounded == actuals_a)
        & (predicted_score_b_rounded == actuals_b)
    )
    exact_score_accuracy = (exact_matches / len(actuals_a)) * 100 if actuals_a else 0

    correct_outcomes = 0
    for i in range(len(actuals_a)):
        actual_outcome = (
            1
            if actuals_a[i] > actuals_b[i]
            else (-1 if actuals_a[i] < actuals_b[i] else 0)
        )
        predicted_outcome = (
            1
            if predicted_lambda_a[i] > predicted_lambda_b[i]
            else (-1 if predicted_lambda_a[i] < predicted_lambda_b[i] else 0)
        )

        if actual_outcome == predicted_outcome:
            correct_outcomes += 1

    outcome_accuracy = (correct_outcomes / len(actuals_a)) * 100 if actuals_a else 0

    print("\n===== VÝSLEDKY TESTU PŘESNOSTI MODELU =====")
    print(f"Počet testovaných zápasů: {len(actuals_a)}")
    print(f"Průměrná absolutní chyba (MAE) - Skóre A: {mae_a:.3f}")
    print(f"Průměrná absolutní chyba (MAE) - Skóre B: {mae_b:.3f}")
    print(f"Přesnost predikce přesného skóre: {exact_score_accuracy:.2f}%")
    print(
        f"Přesnost predikce výsledku (Výhra A/Výhra B/Remíza): {outcome_accuracy:.2f}%"
    )
    print("===========================================\n")


if __name__ == "__main__":
    test_prediction_accuracy()
