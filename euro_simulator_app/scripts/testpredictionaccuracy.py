import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

# Funkce pro aktualizaci Elo hodnocení
def update_elo(team_a_rating, team_b_rating, score_a, score_b, is_home_a=False, is_neutral=True, tournament_weight=50):
    diff = team_a_rating - team_b_rating
    if is_home_a:
        diff += 100
    elif not is_neutral:
        diff -= 100
    we_a = 1 / (10 ** (-diff / 400) + 1)
    we_b = 1 - we_a
    w_a = 1 if score_a > score_b else 0 if score_a < score_b else 0.5
    w_b = 1 - w_a
    goal_diff = abs(score_a - score_b)
    g = 1 if goal_diff <= 1 else 1.5 if goal_diff == 2 else (11 + goal_diff) / 8
    delta_a = tournament_weight * g * (w_a - we_a)
    delta_b = tournament_weight * g * (w_b - we_b)
    return delta_a, delta_b, team_a_rating + delta_a, team_b_rating + delta_b

# Načtení trénovacích dat
try:
    training_data = pd.read_csv('vsechny_evropske_zapasy.csv')
except FileNotFoundError:
    print("Soubor 'vsechny_evropske_zapasy.csv' nebyl nalezen. Vytvořte ho, prosím.")
    exit()

# Načtení skutečných výsledků
try:
    real_results = pd.read_csv('realne_vysledky.csv')
except FileNotFoundError:
    print("Soubor 'realne_vysledky.csv' nebyl nalezen. Vytvořte ho, prosím.")
    exit()

# Příprava dat pro model
X_train = training_data.copy()
X_train["PrevRatingA"] = X_train["NewRatingA"] - X_train["RatingChangeA"]
X_train["PrevRatingB"] = X_train["NewRatingB"] - X_train["RatingChangeB"]
X_train["HomeAdvantage"] = (X_train["Location"] == X_train["TeamA"]).astype(int)
X_train["NeutralGround"] = ((X_train["Location"] != X_train["TeamA"]) & 
                           (X_train["Location"] != X_train["TeamB"])).astype(int)
X_train["IsFriendly"] = X_train["Tournament"].str.contains("Friendly").astype(int)
X_train["IsQualifier"] = X_train["Tournament"].str.contains("qualifier").astype(int)
main_tournaments = ["Confederations Cup", "European Championship", "European Nations League", 
                   "Intercontinental Championship", "King's Cup", "World Cup"]
X_train["IsMainTournament"] = X_train["Tournament"].isin(main_tournaments).astype(int)

# Trénování modelů pro skóre
features = ["PrevRatingA", "PrevRatingB", "HomeAdvantage", "NeutralGround",
            "IsFriendly", "IsQualifier", "IsMainTournament"]
X = X_train[features]
y_scoreA = X_train["ScoreA"]
y_scoreB = X_train["ScoreB"]

model_scoreA = PoissonRegressor(alpha=0.001, max_iter=2000)
model_scoreB = PoissonRegressor(alpha=0.001, max_iter=2000)
model_scoreA.fit(X, y_scoreA)
model_scoreB.fit(X, y_scoreB)

# Počáteční hodnocení týmů
team_ratings = {
    "Germany": 2059, "Scotland": 1773, "Hungary": 1709, "Switzerland": 1847, "Spain": 2044,
    "Croatia": 1872, "Italy": 1960, "Albania": 1749, "Slovenia": 1695, "Denmark": 1852,
    "Serbia": 1827, "England": 2088, "Poland": 1832, "Netherlands": 1975, "Austria": 1813,
    "France": 2063, "Romania": 1616, "Ukraine": 1813, "Belgium": 1975, "Slovakia": 1700,
    "Georgia": 1641, "Czechia": 1749, "Turkey": 1827, "Portugal": 2009
}

# Funkce pro simulaci zápasu
def simulate_match(team_a, team_b, location, tournament, team_ratings, model_a, model_b, features):
    match_data = pd.DataFrame({
        "TeamA": [team_a],
        "TeamB": [team_b],
        "Location": [location],
        "Tournament": [tournament],
        "PrevRatingA": [team_ratings[team_a]],
        "PrevRatingB": [team_ratings[team_b]],
        "HomeAdvantage": [(location == team_a)],
        "NeutralGround": [(location != team_a) and (location != team_b)],
        "IsFriendly": [0],
        "IsQualifier": [0],
        "IsMainTournament": [1]
    })
    
    X = match_data[features]
    
    # Použití Poissonova rozdělení pro simulaci gólů (stejná metoda jako v tvém kódu)
    """noiseA = np.random.normal(0, 0.2)  # Šum pro tým A
    noiseB = np.random.normal(0, 0.2)  # Šum pro tým B

    score_a = np.round(model_a.predict(X)[0] + noiseA).astype(int).clip(0)
    score_b = np.round(model_b.predict(X)[0] + noiseB).astype(int).clip(0)"""

    score_a = np.random.poisson(max(0, model_a.predict(X)[0]))
    score_b = np.random.poisson(max(0, model_b.predict(X)[0]))
    
    tournament_weight = 50  # Pro skupinovou fázi
    delta_a, delta_b, new_rating_a, new_rating_b = update_elo(
        team_ratings[team_a], team_ratings[team_b], score_a, score_b,
        is_home_a=(location == team_a), is_neutral=(location != team_a and location != team_b),
        tournament_weight=tournament_weight
    )
    
    team_ratings[team_a] = new_rating_a
    team_ratings[team_b] = new_rating_b
    
    return {"team_a": team_a, "team_b": team_b, "score_a": score_a, "score_b": score_b}

# Funkce pro spuštění simulace skupinové fáze a porovnání s reálnými výsledky
def run_group_stage_simulation(n_simulations=10000):
    mae_scores = []
    outcome_accuracies = []
    exact_scores = []
    
    for iteration in range(n_simulations):
        print(f"Simulace {iteration+1}/{n_simulations}")
        sim_ratings = team_ratings.copy()
        predictions = []
        
        # Simulace zápasů podle reálných výsledků
        for _, row in real_results.iterrows():
            team_a = row["TeamA"]
            team_b = row["TeamB"]
            location = row["Location"]
            tournament = "European Championship"
            
            result = simulate_match(team_a, team_b, location, tournament, sim_ratings, model_scoreA, model_scoreB, features)
            predictions.append({
                "PredA": result["score_a"],
                "PredB": result["score_b"],
                "RealA": row["RealA"],
                "RealB": row["RealB"]
            })
        
        # Výpočet metrik přesnosti
        df = pd.DataFrame(predictions)
        mae = (mean_absolute_error(df["RealA"], df["PredA"]) + 
               mean_absolute_error(df["RealB"], df["PredB"])) / 2
        mae_scores.append(mae)
        
        correct_outcomes = 0
        for _, r in df.iterrows():
            real_outcome = np.sign(r["RealA"] - r["RealB"])
            pred_outcome = np.sign(r["PredA"] - r["PredB"])
            correct_outcomes += (real_outcome == pred_outcome)
        outcome_accuracies.append(correct_outcomes / len(df))
        
        exact = ((df["PredA"] == df["RealA"]) & (df["PredB"] == df["RealB"])).mean()
        exact_scores.append(exact)
    
    # Výpis výsledků
    print(f"\nPrůměrná MAE: {np.mean(mae_scores):.2f}")
    print(f"Průměrná přesnost výsledků: {np.mean(outcome_accuracies):.1%}")
    print(f"Průměrná přesnost přesného skóre: {np.mean(exact_scores):.1%}")
    print(f"Nejlepší MAE: {np.min(mae_scores):.2f}")
    print(f"Nejhorší MAE: {np.max(mae_scores):.2f}")
    print(f"Nejpřesnější simulace (výsledek): {np.max(outcome_accuracies):.1%}")
    print(f"Nejpřesnější simulace (přesné skóre): {np.max(exact_scores):.1%}")

# Spuštění simulace
print("Probíhá simulace skupinové fáze Euro...")
run_group_stage_simulation(10000)