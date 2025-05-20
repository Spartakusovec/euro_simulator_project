import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
import sqlite3
import os
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATABASE_PATH = os.path.join(BASE_DIR, "data", "database.db")
TRAINING_DATA_CSV = os.path.join(BASE_DIR, "data", "vsechny_evropske_zapasy.csv")
ELO_CONSTANT = 400
BASE_TOURNAMENT_WEIGHT = 50
MAIN_TOURNAMENT_WEIGHT = 100
HOME_ADVANTAGE = 100


def update_elo(
    team_a_rating,
    team_b_rating,
    score_a,
    score_b,
    is_home_a=False,
    is_neutral=True,
    tournament_weight=50,
):
    diff = team_a_rating - team_b_rating
    if is_home_a:
        diff += HOME_ADVANTAGE
    elif not is_neutral:
        diff -= HOME_ADVANTAGE
    we_a = 1 / (10 ** (-diff / ELO_CONSTANT) + 1)
    we_b = 1 - we_a
    w_a = 1 if score_a > score_b else 0 if score_a < score_b else 0.5
    w_b = 1 - w_a
    goal_diff = abs(score_a - score_b)
    g = 1 if goal_diff <= 1 else 1.5 if goal_diff == 2 else (11 + goal_diff) / 8
    delta_a = tournament_weight * g * (w_a - we_a)
    delta_b = tournament_weight * g * (w_b - we_b)
    return delta_a, delta_b, team_a_rating + delta_a, team_b_rating + delta_b


def calculate_initial_elo_from_rosters(db_path, new_min_elo=1616, new_max_elo=2088):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT nationality, SUM(overall_rating_in_position) as total_team_rating FROM rosters GROUP BY nationality"
        df_team_ratings = pd.read_sql_query(query, conn)

        if df_team_ratings.empty:
            logging.warning("No data found in rosters table to calculate initial ELOs.")
            return {}

        if 'nationality' in df_team_ratings.columns:
            df_team_ratings['nationality'] = df_team_ratings['nationality'].replace('Türkiye', 'Turkey')

        df_team_ratings['total_team_rating'] = pd.to_numeric(df_team_ratings['total_team_rating'], errors='coerce')
        df_team_ratings.dropna(subset=['total_team_rating'], inplace=True)

        if df_team_ratings.empty:
            logging.warning("No valid numeric total_team_rating found after conversion in rosters data.")
            return {}

        min_overall_rating = df_team_ratings['total_team_rating'].min()
        max_overall_rating = df_team_ratings['total_team_rating'].max()

        normalized_elos = {}
        if min_overall_rating == max_overall_rating:
            logging.info("All teams in rosters have the same overall rating. Assigning mid-range ELO.")
            avg_elo = round((new_min_elo + new_max_elo) / 2)
            for index, row in df_team_ratings.iterrows():
                normalized_elos[row['nationality']] = avg_elo
        else:
            for index, row in df_team_ratings.iterrows():
                nationality = row['nationality']
                total_rating = row['total_team_rating']
                normalized_elo = new_min_elo + (
                    (total_rating - min_overall_rating) * (new_max_elo - new_min_elo) /
                    (max_overall_rating - min_overall_rating)
                )
                normalized_elos[nationality] = round(normalized_elo)
        
        logging.info(f"Successfully calculated ELOs from rosters for {len(normalized_elos)} teams.")
        return normalized_elos

    except sqlite3.Error as e:
        logging.error(f"Database error while calculating initial ELOs: {e}")
        return {}
    except Exception as e:
        logging.error(f"General error while calculating initial ELOs: {e}")
        return {}
    finally:
        if conn:
            conn.close()


try:
    training_data = pd.read_csv(TRAINING_DATA_CSV)
except FileNotFoundError:
    print(f"Soubor '{TRAINING_DATA_CSV}' nebyl nalezen.")
    exit()
except Exception as e:
    logging.error(f"Chyba při načítání trénovacích dat: {e}")
    exit()
X_train = training_data.copy()
X_train["PrevRatingA"] = X_train["NewRatingA"] - X_train["RatingChangeA"]
X_train["PrevRatingB"] = X_train["NewRatingB"] - X_train["RatingChangeB"]
X_train["HomeAdvantage"] = (X_train["Location"] == X_train["TeamA"]).astype(int)
X_train["NeutralGround"] = (
    (X_train["Location"] != X_train["TeamA"])
    & (X_train["Location"] != X_train["TeamB"])
).astype(int)
X_train["IsFriendly"] = (
    X_train["Tournament"].str.contains("Friendly", case=False, na=False).astype(int)
)
X_train["IsQualifier"] = (
    X_train["Tournament"].str.contains("qualifier", case=False, na=False).astype(int)
)
main_tournaments = [
    "Confederations Cup",
    "European Championship",
    "European Nations League",
    "Intercontinental Championship",
    "King's Cup",
    "World Cup",
]
X_train["IsMainTournament"] = X_train["Tournament"].isin(main_tournaments).astype(int)
features = [
    "PrevRatingA",
    "PrevRatingB",
    "HomeAdvantage",
    "NeutralGround",
    "IsFriendly",
    "IsQualifier",
    "IsMainTournament",
]
logging.info("Trénování modelů Poissonovy regrese...")
try:
    model_scoreA = PoissonRegressor(alpha=0.1, max_iter=2000)
    model_scoreB = PoissonRegressor(alpha=0.1, max_iter=2000)
    model_scoreA.fit(X_train[features], X_train["ScoreA"])
    model_scoreB.fit(X_train[features], X_train["ScoreB"])
    logging.info("Modely úspěšně natrénovány.")
except Exception as e:
    logging.error(f"Chyba při trénování modelů: {e}")
    exit()

default_euro_team_ratings = {
    "Albania": 1651, "Austria": 1805, "Belgium": 1939, "Croatia": 1850,
    "Czechia": 1735, "Denmark": 1840, "England": 2088, "France": 2068,
    "Georgia": 1641, "Germany": 2033, "Hungary": 1676, "Italy": 1949,
    "Netherlands": 1959, "Poland": 1825, "Portugal": 1994, "Romania": 1616,
    "Scotland": 1765, "Serbia": 1810, "Slovakia": 1681, "Slovenia": 1686,
    "Spain": 2028, "Switzerland": 1845, "Turkey": 1810, "Ukraine": 1800,
}

logging.info("Attempting to calculate initial ELO ratings from rosters...")
calculated_roster_elos = calculate_initial_elo_from_rosters(DATABASE_PATH, new_min_elo=1616, new_max_elo=2088)

team_ratings = default_euro_team_ratings.copy()

if calculated_roster_elos:
    updated_count = 0
    missing_from_rosters_count = 0
    for team_name in team_ratings.keys(): 
        if team_name in calculated_roster_elos:
            team_ratings[team_name] = calculated_roster_elos[team_name]
            updated_count += 1
        else:
            logging.warning(f"Team {team_name} is a Euro participant but not found in roster-calculated ELOs. Using default ELO: {team_ratings[team_name]}.")
            missing_from_rosters_count +=1
            
    logging.info(f"Updated ELO ratings for {updated_count} Euro teams from rosters data.")
    if missing_from_rosters_count > 0:
        logging.info(f"{missing_from_rosters_count} Euro teams used default ELOs as they were not in roster calculations.")
else:
    logging.warning("Failed to calculate ELOs from rosters or rosters table was empty/invalid. Using default ELO ratings for all Euro teams.")

groups = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Slovenia", "Denmark", "Serbia", "England"],
    "D": ["Poland", "Netherlands", "Austria", "France"],
    "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "F": ["Turkey", "Georgia", "Portugal", "Czechia"],
}

all_participating_teams = set(team for group_teams in groups.values() for team in group_teams)
for team_name_in_group in all_participating_teams:
    if team_name_in_group not in team_ratings:
        logging.error(f"CRITICAL ERROR: Team '{team_name_in_group}' is defined in groups but is missing from final team_ratings. This indicates a setup issue.")
        logging.error("Please check roster data, default ELO list, and group definitions. Exiting.")
        exit()

logging.info(f"Final initial ELO ratings for simulation: {team_ratings}")

group_matches = [
    ["Germany", "Scotland", "European Championship", "Germany"],
    ["Hungary", "Switzerland", "European Championship", "Germany"],
    ["Spain", "Croatia", "European Championship", "Germany"],
    ["Italy", "Albania", "European Championship", "Germany"],
    ["Slovenia", "Denmark", "European Championship", "Germany"],
    ["Serbia", "England", "European Championship", "Germany"],
    ["Poland", "Netherlands", "European Championship", "Germany"],
    ["Austria", "France", "European Championship", "Germany"],
    ["Romania", "Ukraine", "European Championship", "Germany"],
    ["Belgium", "Slovakia", "European Championship", "Germany"],
    ["Turkey", "Georgia", "European Championship", "Germany"],
    ["Portugal", "Czechia", "European Championship", "Germany"],
    ["Germany", "Hungary", "European Championship", "Germany"],
    ["Scotland", "Switzerland", "European Championship", "Germany"],
    ["Croatia", "Albania", "European Championship", "Germany"],
    ["Spain", "Italy", "European Championship", "Germany"],
    ["Slovenia", "Serbia", "European Championship", "Germany"],
    ["Denmark", "England", "European Championship", "Germany"],
    ["Poland", "Austria", "European Championship", "Germany"],
    ["Netherlands", "France", "European Championship", "Germany"],
    ["Slovakia", "Ukraine", "European Championship", "Germany"],
    ["Belgium", "Romania", "European Championship", "Germany"],
    ["Georgia", "Czechia", "European Championship", "Germany"],
    ["Turkey", "Portugal", "European Championship", "Germany"],
    ["Switzerland", "Germany", "European Championship", "Germany"],
    ["Scotland", "Hungary", "European Championship", "Germany"],
    ["Albania", "Spain", "European Championship", "Germany"],
    ["Croatia", "Italy", "European Championship", "Germany"],
    ["England", "Slovenia", "European Championship", "Germany"],
    ["Denmark", "Serbia", "European Championship", "Germany"],
    ["Netherlands", "Austria", "European Championship", "Germany"],
    ["France", "Poland", "European Championship", "Germany"],
    ["Slovakia", "Romania", "European Championship", "Germany"],
    ["Ukraine", "Belgium", "European Championship", "Germany"],
    ["Georgia", "Portugal", "European Championship", "Germany"],
    ["Czechia", "Turkey", "European Championship", "Germany"],
]


def simulate_match(
    team_a, team_b, location, tournament, ratings, model_a, model_b, knockout=False
):
    match_features = {
        "PrevRatingA": [ratings[team_a]],
        "PrevRatingB": [ratings[team_b]],
        "HomeAdvantage": [location == team_a],
        "NeutralGround": [(location != team_a) and (location != team_b)],
        "IsFriendly": [0],
        "IsQualifier": [0],
        "IsMainTournament": [1],
    }
    X = pd.DataFrame(match_features)
    lambda_a = max(0.01, model_a.predict(X[features])[0])
    lambda_b = max(0.01, model_b.predict(X[features])[0])
    score_a = np.random.poisson(lambda_a)
    score_b = np.random.poisson(lambda_b)
    if knockout and score_a == score_b:
        if np.random.rand() > 0.5:
            score_a += 1
        else:
            score_b += 1

    weight = MAIN_TOURNAMENT_WEIGHT if "Final" in tournament else BASE_TOURNAMENT_WEIGHT
    delta_a, delta_b, new_a, new_b = update_elo(
        ratings[team_a],
        ratings[team_b],
        score_a,
        score_b,
        is_home_a=(location == team_a),
        is_neutral=((location != team_a) and (location != team_b)),
        tournament_weight=weight,
    )

    ratings[team_a] = new_a
    ratings[team_b] = new_b

    return score_a, score_b, new_a, new_b


def calculate_group_standings(matches, group_teams):
    standings = {team: {"Points": 0, "GF": 0, "GA": 0, "GD": 0} for team in group_teams}
    for score_a, score_b, team_a, team_b in matches:
        if team_a not in standings or team_b not in standings:
            continue
        standings[team_a]["GF"] += score_a
        standings[team_a]["GA"] += score_b
        standings[team_b]["GF"] += score_b
        standings[team_b]["GA"] += score_a
        if score_a > score_b:
            standings[team_a]["Points"] += 3
        elif score_a < score_b:
            standings[team_b]["Points"] += 3
        else:
            standings[team_a]["Points"] += 1
            standings[team_b]["Points"] += 1
    for team in group_teams:
        if team in standings:
            standings[team]["GD"] = standings[team]["GF"] - standings[team]["GA"]
    return sorted(
        standings.items(), key=lambda x: (-x[1]["Points"], -x[1]["GD"], -x[1]["GF"])
    )


def get_r16_pairings(qualified_teams, third_place_info):
    if len(qualified_teams) != 16 or len(third_place_info) != 4:
        logging.error(
            f"R16 PAIRING ERROR: Incorrect number of teams. Qualified ({len(qualified_teams)}): {qualified_teams}, 3rd Place Info ({len(third_place_info)}): {third_place_info}"
        )
        return None

    try:
        third_place_map = {info[2]: info[0] for info in third_place_info}
        if len(third_place_map) != 4:
            logging.error(
                f"R16 PAIRING ERROR: third_place_map has incorrect size ({len(third_place_map)}). Expected 4 unique groups. Map: {third_place_map}, Input Info: {third_place_info}"
            )
            return None
    except (IndexError, TypeError) as e:
        logging.error(
            f"R16 PAIRING ERROR: Failed to create third_place_map from third_place_info. Error: {e}. Info: {third_place_info}"
        )
        return None

    third_groups = tuple(sorted(third_place_map.keys()))

    try:
        t1A = qualified_teams[0]
        t2A = qualified_teams[1]
        t1B = qualified_teams[2]
        t2B = qualified_teams[3]
        t1C = qualified_teams[4]
        t2C = qualified_teams[5]
        t1D = qualified_teams[6]
        t2D = qualified_teams[7]
        t1E = qualified_teams[8]
        t2E = qualified_teams[9]
        t1F = qualified_teams[10]
        t2F = qualified_teams[11]
    except IndexError:
        logging.error(
            f"R16 PAIRING ERROR: qualified_teams list too short ({len(qualified_teams)}). List: {qualified_teams}"
        )
        return None

    third_place_opponents_rules = {
        ("A", "B", "C", "D"): {"1B": "3A", "1C": "3B", "1E": "3D", "1F": "3C"},
        ("A", "B", "C", "E"): {"1B": "3A", "1C": "3E", "1E": "3B", "1F": "3C"},
        ("A", "B", "C", "F"): {"1B": "3A", "1C": "3F", "1E": "3C", "1F": "3B"},
        ("A", "B", "D", "E"): {"1B": "3A", "1C": "3E", "1E": "3D", "1F": "3B"},
        ("A", "B", "D", "F"): {"1B": "3A", "1C": "3F", "1E": "3D", "1F": "3B"},
        ("A", "B", "E", "F"): {"1B": "3A", "1C": "3F", "1E": "3B", "1F": "3E"},
        ("A", "C", "D", "E"): {"1B": "3A", "1C": "3E", "1E": "3D", "1F": "3C"},
        ("A", "C", "D", "F"): {"1B": "3A", "1C": "3F", "1E": "3D", "1F": "3C"},
        ("A", "C", "E", "F"): {"1B": "3A", "1C": "3F", "1E": "3E", "1F": "3C"},
        ("A", "D", "E", "F"): {"1B": "3A", "1C": "3F", "1E": "3E", "1F": "3D"},
        ("B", "C", "D", "E"): {"1B": "3D", "1C": "3E", "1E": "3B", "1F": "3C"},
        ("B", "C", "D", "F"): {"1B": "3D", "1C": "3F", "1E": "3B", "1F": "3C"},
        ("B", "C", "E", "F"): {"1B": "3E", "1C": "3F", "1E": "3B", "1F": "3C"},
        ("B", "D", "E", "F"): {"1B": "3E", "1C": "3F", "1E": "3B", "1F": "3D"},
        ("C", "D", "E", "F"): {"1B": "3E", "1C": "3F", "1E": "3C", "1F": "3D"},
    }

    if third_groups not in third_place_opponents_rules:
        logging.error(
            f"R16 PAIRING ERROR: Invalid combination of third-place groups: {third_groups}. Map: {third_place_map}"
        )
        return None

    specific_opponents = third_place_opponents_rules[third_groups]

    try:

        opp_1B = third_place_map[specific_opponents["1B"][1]]
        opp_1C = third_place_map[specific_opponents["1C"][1]]
        opp_1E = third_place_map[specific_opponents["1E"][1]]
        opp_1F = third_place_map[specific_opponents["1F"][1]]
    except KeyError as e:
        logging.error(
            f"R16 PAIRING ERROR: KeyError finding opponent team. Missing group key: {e}. third_groups: {third_groups}, rules: {specific_opponents}, third_place_map: {third_place_map}"
        )
        return None

    r16_matches = [
        [t1B, opp_1B],  # Match 37 (Index 0)
        [t1A, t2C],  # Match 38 (Index 1)
        [t1F, opp_1F],  # Match 39 (Index 2)
        [t2D, t2E],  # Match 40 (Index 3)
        [t1E, opp_1E],  # Match 41 (Index 4)
        [t1D, t2F],  # Match 42 (Index 5)
        [t1C, opp_1C],  # Match 43 (Index 6)
        [t2A, t2B],  # Match 44 (Index 7)
    ]

    logging.debug(
        f"Determined R16 Pairings for third-groups {third_groups}: {r16_matches}"
    )
    return r16_matches


def run_simulation(n_simulations=10000):
    winners = {team: 0 for team in team_ratings}
    finalists = {team: 0 for team in team_ratings}
    semifinalists = {team: 0 for team in team_ratings}
    quarterfinalists = {team: 0 for team in team_ratings}

    first_simulation_details = None

    for sim in tqdm(range(n_simulations), desc="Simulace turnaje"):
        ratings = team_ratings.copy()
        group_results = []

        match_details = [] if sim == 0 else None  # Text descriptions of matches
        knockout_results = [] if sim == 0 else None  # Text descriptions of KO matches
        group_standings_dict = {} if sim == 0 else None  # Group standings for sim 0
        per_match_elo_snapshots = [] if sim == 0 else None
        match_counter = 0 if sim == 0 else -1

        if sim == 0:
            per_match_elo_snapshots.append(
                {
                    "match_order": match_counter,
                    "stage": "Initial",
                    "match_description": "Initial Ratings",
                    "team_elos": ratings.copy(),  # Snapshot of all initial ratings
                }
            )

        for team_a, team_b, tournament, location in group_matches:
            score_a, score_b, elo_a_after, elo_b_after = simulate_match(
                team_a,
                team_b,
                location,
                tournament,
                ratings,
                model_scoreA,
                model_scoreB,
            )
            group_results.append((score_a, score_b, team_a, team_b))

            if sim == 0:
                match_counter += 1
                match_desc_text = f"{team_a} {score_a}-{score_b} {team_b} (Elo: {elo_a_after:.0f} vs {elo_b_after:.0f})"
                match_details.append(match_desc_text)

                per_match_elo_snapshots.append(
                    {
                        "match_order": match_counter,
                        "stage": "Group",
                        "match_description": f"{team_a} vs {team_b}",
                        "team_elos": {
                            team_a: elo_a_after,
                            team_b: elo_b_after,
                        },
                    }
                )

        qualified = []
        third_place = []
        for group_name, teams_in_group in groups.items():
            group_games = [
                (s_a, s_b, t_a, t_b)
                for s_a, s_b, t_a, t_b in group_results
                if t_a in teams_in_group and t_b in teams_in_group
            ]
            standings = calculate_group_standings(group_games, teams_in_group)
            if sim == 0:
                group_standings_dict[group_name] = standings
            if standings:
                qualified.append(standings[0][0])
                if len(standings) > 1:
                    qualified.append(standings[1][0])
                if len(standings) > 2:
                    third_place.append((standings[2][0], standings[2][1], group_name))
        third_place.sort(key=lambda x: (-x[1]["Points"], -x[1]["GD"], -x[1]["GF"]))
        best_third_info = third_place[:4]
        best_third_teams = [x[0] for x in best_third_info]
        qualified.extend(best_third_teams)

        if len(qualified) != 16:
            logging.warning(
                f"Sim {sim}: Incorrect number of qualified teams ({len(qualified)}). Skipping KO stage."
            )
            if sim == 0:
                first_simulation_details = {
                    "error": f"Kvalifikováno pouze {len(qualified)} týmů.",
                    "per_match_elo_snapshots": per_match_elo_snapshots,
                }
            continue

        if sim == 0:
            third_place_details_for_dict = [
                (x[0], x[2], x[1]["Points"], x[1]["GD"], x[1]["GF"])
                for x in third_place
            ]
            first_simulation_details = {
                "match_results": match_details,
                "group_standings": group_standings_dict,
                "third_place_teams": third_place_details_for_dict,
                "qualified_teams": qualified[:],
                "knockout_results": knockout_results,
                "final_ratings": None,
                "per_match_elo_snapshots": per_match_elo_snapshots,
            }

        try:
            r16_matches = get_r16_pairings(qualified, best_third_info)
            if r16_matches is None:
                logging.error(
                    f"Sim {sim}: Failed to determine R16 pairings (returned None). Skipping KO stage."
                )
                if sim == 0 and first_simulation_details:
                    first_simulation_details["error"] = (
                        "Nepodařilo se určit dvojice pro osmifinále."
                    )
                continue

            r16_winners = [None] * 8
            stage_name = "R16"
            for i, (a, b) in enumerate(r16_matches):
                if a not in ratings or b not in ratings:
                    logging.error(
                        f"Sim {sim}, R16 Match {i+37}: Team not found in ratings - {a} or {b}. Qualified: {qualified}"
                    )
                    raise KeyError(f"Team not found: {a if a not in ratings else b}")

                s_a, s_b, elo_a_after, elo_b_after = simulate_match(
                    a,
                    b,
                    "Germany",
                    f"European Championship {stage_name}",
                    ratings,
                    model_scoreA,
                    model_scoreB,
                    knockout=True,
                )
                winner = a if s_a > s_b else b
                r16_winners[i] = winner
                quarterfinalists[a] += 1
                quarterfinalists[b] += 1

                if sim == 0:
                    match_counter += 1
                    knockout_results.append(
                        f"{stage_name} (M{i+37}): {a} {s_a}-{s_b} {b} (Winner: {winner}, Elo: {elo_a_after:.0f} vs {elo_b_after:.0f})"
                    )
                    first_simulation_details["per_match_elo_snapshots"].append(
                        {
                            "match_order": match_counter,
                            "stage": stage_name,
                            "match_description": f"M{i+37}: {a} vs {b}",
                            "team_elos": {a: elo_a_after, b: elo_b_after},
                        }
                    )

            qf_matches = [
                [r16_winners[2], r16_winners[0]],  # QF1
                [r16_winners[4], r16_winners[5]],  # QF2
                [r16_winners[6], r16_winners[7]],  # QF3
                [r16_winners[3], r16_winners[1]],  # QF4
            ]
            # --- Simulace čtvrtfinále ---
            qf_winners = [None] * 4
            stage_name = "QF"
            qf_match_numbers = ["QF1", "QF2", "QF3", "QF4"]
            for i, (a, b) in enumerate(qf_matches):
                s_a, s_b, elo_a_after, elo_b_after = simulate_match(
                    a,
                    b,
                    "Germany",
                    f"European Championship {stage_name}",
                    ratings,
                    model_scoreA,
                    model_scoreB,
                    knockout=True,
                )
                winner = a if s_a > s_b else b
                qf_winners[i] = winner
                semifinalists[a] += 1
                semifinalists[b] += 1
                if sim == 0:
                    match_counter += 1
                    qf_num = qf_match_numbers[i]
                    first_simulation_details["knockout_results"].append(
                        f"{stage_name} ({qf_num}): {a} {s_a}-{s_b} {b} (Winner: {winner}, Elo: {elo_a_after:.0f} vs {elo_b_after:.0f})"
                    )
                    first_simulation_details["per_match_elo_snapshots"].append(
                        {
                            "match_order": match_counter,
                            "stage": stage_name,
                            "match_description": f"{qf_num}: {a} vs {b}",
                            "team_elos": {a: elo_a_after, b: elo_b_after},
                        }
                    )

            # --- Semifinále ---

            sf_matches = [
                [qf_winners[0], qf_winners[1]],
                [qf_winners[2], qf_winners[3]],
            ]
            final_teams = [None] * 2
            stage_name = "SF"
            sf_match_numbers = ["SF1", "SF2"]
            for i, (a, b) in enumerate(sf_matches):
                s_a, s_b, elo_a_after, elo_b_after = simulate_match(
                    a,
                    b,
                    "Germany",
                    f"European Championship {stage_name}",
                    ratings,
                    model_scoreA,
                    model_scoreB,
                    knockout=True,
                )
                winner = a if s_a > s_b else b
                final_teams[i] = winner
                finalists[a] += 1
                finalists[b] += 1
                if sim == 0:
                    match_counter += 1
                    sf_num = sf_match_numbers[i]
                    first_simulation_details["knockout_results"].append(
                        f"{stage_name} ({sf_num}): {a} {s_a}-{s_b} {b} (Winner: {winner}, Elo: {elo_a_after:.0f} vs {elo_b_after:.0f})"
                    )
                    first_simulation_details["per_match_elo_snapshots"].append(
                        {
                            "match_order": match_counter,
                            "stage": stage_name,
                            "match_description": f"{sf_num}: {a} vs {b}",
                            "team_elos": {a: elo_a_after, b: elo_b_after},
                        }
                    )

            # --- Finále  ---
            team_a = final_teams[0]
            team_b = final_teams[1]
            stage_name = "Final"
            s_a, s_b, elo_a_after, elo_b_after = simulate_match(
                team_a,
                team_b,
                "Germany",
                f"European Championship {stage_name}",
                ratings,
                model_scoreA,
                model_scoreB,
                knockout=True,
            )
            winner = team_a if s_a > s_b else team_b
            winners[winner] += 1
            if sim == 0:
                match_counter += 1
                first_simulation_details["knockout_results"].append(
                    f"{stage_name}: {team_a} {s_a}-{s_b} {team_b} (Winner: {winner}, Elo: {elo_a_after:.0f} vs {elo_b_after:.0f})"
                )
                first_simulation_details["per_match_elo_snapshots"].append(
                    {
                        "match_order": match_counter,
                        "stage": stage_name,
                        "match_description": f"{team_a} vs {team_b}",
                        "team_elos": {team_a: elo_a_after, team_b: elo_b_after},
                    }
                )
                first_simulation_details["final_ratings"] = ratings.copy()

        except (IndexError, ValueError, KeyError) as e:
            logging.error(f"Sim {sim}: Chyba při zpracování KO fáze: {e}")
            logging.exception("Detail:")
            if (
                sim == 0
                and first_simulation_details
                and "error" not in first_simulation_details
            ):
                first_simulation_details["error"] = f"Chyba KO fáze: {e}"
            continue
        except Exception as e:
            logging.error(f"Sim {sim}: Neočekávaná chyba KO fáze: {e}")
            logging.exception("Detail:")
            if (
                sim == 0
                and first_simulation_details
                and "error" not in first_simulation_details
            ):
                first_simulation_details["error"] = f"Chyba KO fáze: {e}"
            continue

    total_sims_run = n_simulations
    win_probs = {t: (c / total_sims_run) * 100 for t, c in winners.items()}
    final_probs = {t: (c / total_sims_run) * 100 for t, c in finalists.items()}
    semi_probs = {t: (c / total_sims_run) * 100 for t, c in semifinalists.items()}
    quarter_probs = {t: (c / total_sims_run) * 100 for t, c in quarterfinalists.items()}

    return win_probs, final_probs, semi_probs, quarter_probs, first_simulation_details


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"DB: Úspěšně připojeno k {db_file}")
    except sqlite3.Error as e:
        logging.error(f"DB: Chyba při připojování: {e}")
    return conn


def save_probabilities_to_db(conn, win_p, final_p, semi_p, quarter_p):
    if conn is None:
        logging.error("DB: Nelze uložit pravděpodobnosti: Chybí připojení.")
        return
    try:
        table_name = "simulation_probabilities"
        logging.info(f"DB: Příprava pravděpodobností pro '{table_name}'...")
        all_teams = list(team_ratings.keys())
        results_list = [
            {
                "nationality": team,
                "win_prob": win_p.get(team, 0),
                "final_prob": final_p.get(team, 0),
                "semi_prob": semi_p.get(team, 0),
                "quarter_prob": quarter_p.get(team, 0),
            }
            for team in all_teams
        ]
        results_df = pd.DataFrame(results_list)
        logging.info(f"DB: Ukládání pravděpodobností do '{table_name}'...")
        results_df.to_sql(table_name, conn, if_exists="replace", index=False)
        logging.info(f"DB: Tabulka '{table_name}' naplněna.")
    except Exception as e:
        logging.error(f"DB: Chyba při ukládání pravděpodobností: {e}")
        logging.exception("Detail:")


def save_first_run_details_to_db(conn, details_dict, simulation_id=1):
    if conn is None:
        logging.error("DB: Cannot save details: Connection is missing.")
        return
    if not details_dict or not isinstance(details_dict, dict):
        logging.warning(f"DB: Details data missing or not a dictionary: {details_dict}")
        return
    details_table_name = "simulation_run_details"
    elo_table_name = "elo_snapshots"
    details_to_insert = []
    elo_snapshots_to_insert = []

    try:
        if "match_results" in details_dict and isinstance(
            details_dict["match_results"], list
        ):
            details_to_insert.extend(
                [
                    (simulation_id, "Group Match", str(desc))
                    for desc in details_dict["match_results"]
                ]
            )

        if "group_standings" in details_dict and isinstance(
            details_dict["group_standings"], dict
        ):
            for group_name, standings in details_dict["group_standings"].items():
                team_standing_parts = []
                try:
                    for i, (team, stats) in enumerate(standings, 1):
                        try:
                            if isinstance(stats, dict) and isinstance(team, str):
                                points = stats.get("Points", 0)
                                gd = stats.get("GD", 0)
                                gf = stats.get("GF", 0)
                                ga = stats.get("GA", 0)
                                if not all(
                                    isinstance(val, (int, float))
                                    for val in [points, gd, gf, ga]
                                ):
                                    raise TypeError(
                                        f"Non-numeric value found in stats: Pts={points}, GD={gd}, GF={gf}, GA={ga}"
                                    )
                                team_part = (
                                    f"{i}.{team.strip()} ({points}b,{gd:+},{gf}-{ga})"
                                )
                                team_standing_parts.append(team_part)
                            else:
                                logging.warning(
                                    f"DB: Skipping invalid standing item in group {group_name}: team={team} (type={type(team)}), stats={stats} (type={type(stats)})"
                                )
                        except Exception as inner_e:
                            logging.error(
                                f"DB: Error formatting standing for team '{team}' in group {group_name}. Stats: {stats}. Error: {inner_e}"
                            )
                            team_standing_parts.append(f"{i}.{team.strip()} (Error)")
                    standing_str = ", ".join(team_standing_parts)
                    details_to_insert.append(
                        (simulation_id, f"{group_name} Standing", standing_str)
                    )
                except Exception as e:
                    logging.warning(
                        f"DB: Could not process standings loop for group {group_name}: {e}"
                    )
            logging.info(f"DB: Finished processing group standings.")

        if "third_place_teams" in details_dict and isinstance(
            details_dict["third_place_teams"], list
        ):
            try:

                ranking_str = ", ".join(
                    [
                        f"{i}.{t[0]}({t[1]},{t[2]}b,{t[3]:+})"
                        for i, t in enumerate(details_dict["third_place_teams"], 1)
                        if len(t) >= 4
                    ]
                )
                details_to_insert.append(
                    (simulation_id, "3rd Place Ranking", ranking_str)
                )
            except Exception as e:
                logging.warning(f"DB: Could not process 3rd place ranking: {e}")

        if (
            "qualified_teams" in details_dict
            and isinstance(details_dict["qualified_teams"], list)
            and len(details_dict["qualified_teams"]) == 16
        ):
            details_to_insert.append(
                (
                    simulation_id,
                    "Best 3rd Qualifiers",
                    ", ".join(details_dict["qualified_teams"][12:]),
                )
            )
            try:
                details_to_insert.append(
                    (
                        simulation_id,
                        "Qualified R16 List",
                        json.dumps(details_dict["qualified_teams"]),
                    )
                )
            except Exception as e:
                logging.error(f"DB: Error saving qualified teams JSON: {e}")

        if "knockout_results" in details_dict and isinstance(
            details_dict["knockout_results"], list
        ):
            for desc in details_dict["knockout_results"]:
                stage = "Knockout"
                desc_str = str(desc)
                if desc_str.startswith("R16"):
                    stage = "R16"
                elif desc_str.startswith("QF"):
                    stage = "QF"
                elif desc_str.startswith("SF"):
                    stage = "SF"
                elif desc_str.startswith("Final"):
                    stage = "Final"
                details_to_insert.append((simulation_id, stage, desc_str))

        if "error" in details_dict:
            details_to_insert.append(
                (simulation_id, "Simulation Error", str(details_dict["error"]))
            )

        if "per_match_elo_snapshots" in details_dict and isinstance(
            details_dict["per_match_elo_snapshots"], list
        ):
            for snapshot in details_dict["per_match_elo_snapshots"]:
                if not isinstance(snapshot, dict):
                    continue
                match_order = snapshot.get("match_order", -1)
                stage = snapshot.get("stage", "Unknown")
                match_desc = snapshot.get("match_description", "N/A")
                team_elos = snapshot.get("team_elos", {})
                if not isinstance(team_elos, dict):
                    continue
                for team, elo in team_elos.items():
                    if elo is not None and isinstance(elo, (int, float)):
                        elo_snapshots_to_insert.append(
                            (
                                simulation_id,
                                match_order,
                                stage,
                                match_desc,
                                team,
                                float(elo),
                            )
                        )
                    else:
                        logging.warning(
                            f"DB: Skipping invalid ELO snapshot entry: SimID={simulation_id}, Order={match_order}, Team={team}, ELO={elo}"
                        )
        else:
            logging.warning(
                "DB: Key 'per_match_elo_snapshots' not found or invalid. No per-match ELO data saved."
            )

        cursor = conn.cursor()
        logging.info(f"DB: Deleting old records for simulation_id={simulation_id}...")
        cursor.execute(
            f"DELETE FROM {details_table_name} WHERE simulation_id = ?",
            (simulation_id,),
        )
        cursor.execute(
            f"DELETE FROM {elo_table_name} WHERE simulation_id = ?", (simulation_id,)
        )

        logging.info(
            f"DB: Inserting {len(details_to_insert)} records into '{details_table_name}'..."
        )
        if details_to_insert:
            cursor.executemany(
                f"INSERT INTO {details_table_name} (simulation_id, stage, description) VALUES (?, ?, ?)",
                details_to_insert,
            )

        logging.info(
            f"DB: Inserting {len(elo_snapshots_to_insert)} records into '{elo_table_name}'..."
        )
        if elo_snapshots_to_insert:
            cursor.executemany(
                f"INSERT INTO {elo_table_name} (simulation_id, match_order, stage, match_description, nationality, elo_after_match) VALUES (?, ?, ?, ?, ?, ?)",
                elo_snapshots_to_insert,
            )

        conn.commit()
        logging.info(
            f"DB: Tables '{details_table_name}' and '{elo_table_name}' populated successfully for simulation_id={simulation_id}."
        )

    except sqlite3.Error as db_err:
        logging.error(
            f"DB: SQLite error during save operation for simulation_id={simulation_id}: {db_err}"
        )
        conn.rollback()
    except Exception as e:
        logging.error(
            f"DB: General error while saving details for simulation_id={simulation_id}: {e}"
        )
        logging.exception("Detail:")
        conn.rollback()


if __name__ == "__main__":
    print("Probíhá simulace turnaje Euro 2024...")
    try:
        win_probs, final_probs, semi_probs, quarter_probs, first_run_details_dict = (
            run_simulation(n_simulations=100000)
        )
        simulation_successful = True
        if first_run_details_dict is None:
            logging.error("Nebyla vrácena data z první simulace.")
            simulation_successful = False
            win_probs = {}
        elif "error" in first_run_details_dict:
            logging.warning(
                f"První simulace zaznamenala chybu: {first_run_details_dict['error']}"
            )
    except Exception as e:
        logging.error(f"Chyba během běhu run_simulation: {e}")
        simulation_successful = False
        win_probs, first_run_details_dict = {}, None

    if simulation_successful and win_probs:
        conn = create_connection(DATABASE_PATH)
        if conn:
            try:
                save_probabilities_to_db(
                    conn, win_probs, final_probs, semi_probs, quarter_probs
                )
                if first_run_details_dict is not None:
                    save_first_run_details_to_db(conn, first_run_details_dict)
                else:
                    logging.warning(
                        "Detaily první simulace nebyly dostupné pro uložení."
                    )
            finally:
                conn.close()
            print("\nVýsledky byly uloženy do databáze data/database.db")
        else:
            logging.error("DB: Nepodařilo se připojit k DB pro uložení.")
    elif not simulation_successful:
        print("\nVýsledky se nepodařilo uložit kvůli chybě v simulaci.")
    else:
        print("\nSimulace proběhla, ale nebyly vypočteny pravděpodobnosti pro uložení.")

    print("\n=== VÝSLEDKY SIMULACE (SHRNUTÍ PRAVDĚPODOBNOSTÍ) ===\n")
    if win_probs:
        print("Top 10 pravděpodobností vítězství:")
        for team, prob in sorted(win_probs.items(), key=lambda x: -x[1])[:10]:
            print(f"{team}: {prob:.1f}%")
    else:
        print("Nebylo možné zobrazit pravděpodobnosti.")

    print("\n=== DETAILY PRVNÍ SIMULACE ===")
    if first_run_details_dict and "error" not in first_run_details_dict:
        if "match_results" in first_run_details_dict:
            print("\nVýsledky zápasů ve skupinách:")
            [print(r) for r in first_run_details_dict["match_results"]]
        if "group_standings" in first_run_details_dict:
            print("\nVýsledky skupin:")
            for group_name, standings in first_run_details_dict[
                "group_standings"
            ].items():
                print(f"\n Skupina {group_name}:")
                [
                    print(f" {i}. {t}: {s.get('Points', 0)}b (GD:{s.get('GD', 0)})")
                    for i, (t, s) in enumerate(standings, 1)
                ]
        if "third_place_teams" in first_run_details_dict:
            print("\nPořadí týmů na 3. místech:")
            print(
                " ".join(
                    [
                        f"{i}.{t[0]}({t[1]},{t[2]}b,{t[3]:+})"
                        for i, t in enumerate(
                            first_run_details_dict["third_place_teams"], 1
                        )
                        if len(t) >= 4
                    ]
                )
            )
        if "qualified_teams" in first_run_details_dict:
            print("\nPostupující týmy:")
            print("1-2:", ", ".join(first_run_details_dict["qualified_teams"][:12]))
            print("3.:", ", ".join(first_run_details_dict["qualified_teams"][12:16]))
        if "knockout_results" in first_run_details_dict:
            print("\nVýsledky vyřazovacích kol:")
            [print(r) for r in first_run_details_dict["knockout_results"]]
        if "final_ratings" in first_run_details_dict:
            print("\nKonečné ELO:")
            [
                print(f" {t}: {r:.0f}")
                for t, r in sorted(
                    first_run_details_dict["final_ratings"].items(), key=lambda x: -x[1]
                )
            ]
    elif first_run_details_dict and "error" in first_run_details_dict:
        print(f"\nPrvní simulace skončila s chybou: {first_run_details_dict['error']}")
    else:
        print("\nNebylo možné zobrazit detaily první simulace.")
