# scripts/simulation.py
# VERZE S OPRAVENOU CHYBOU NA KONCI a bez PRAGMA foreign_keys
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
import sqlite3
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'database.db')
TRAINING_DATA_CSV = os.path.join(BASE_DIR, 'data', 'vsechny_evropske_zapasy.csv')
ELO_CONSTANT = 400; BASE_TOURNAMENT_WEIGHT = 50; MAIN_TOURNAMENT_WEIGHT = 100; HOME_ADVANTAGE = 100

# ==========================================
# >>> update_elo - NO CHANGE <<<
# ==========================================
def update_elo(team_a_rating, team_b_rating, score_a, score_b, is_home_a=False, is_neutral=True, tournament_weight=50):
    # --- NO CHANGES HERE ---
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

# --- DATA LOADING & MODEL TRAINING - NO CHANGES ---
try: training_data = pd.read_csv(TRAINING_DATA_CSV)
except FileNotFoundError: print(f"Soubor '{TRAINING_DATA_CSV}' nebyl nalezen."); exit()
except Exception as e: logging.error(f"Chyba při načítání trénovacích dat: {e}"); exit()
X_train = training_data.copy()
X_train["PrevRatingA"] = X_train["NewRatingA"] - X_train["RatingChangeA"]; X_train["PrevRatingB"] = X_train["NewRatingB"] - X_train["RatingChangeB"]
X_train["HomeAdvantage"] = (X_train["Location"] == X_train["TeamA"]).astype(int); X_train["NeutralGround"] = ((X_train["Location"] != X_train["TeamA"]) & (X_train["Location"] != X_train["TeamB"])).astype(int)
X_train["IsFriendly"] = X_train["Tournament"].str.contains("Friendly", case=False, na=False).astype(int); X_train["IsQualifier"] = X_train["Tournament"].str.contains("qualifier", case=False, na=False).astype(int)
main_tournaments = ["Confederations Cup", "European Championship", "European Nations League", "Intercontinental Championship", "King's Cup", "World Cup"]
X_train["IsMainTournament"] = X_train["Tournament"].isin(main_tournaments).astype(int)
features = ["PrevRatingA", "PrevRatingB", "HomeAdvantage", "NeutralGround", "IsFriendly", "IsQualifier", "IsMainTournament"]
logging.info("Trénování modelů Poissonovy regrese...")
try:
    model_scoreA = PoissonRegressor(alpha=0.001, max_iter=2000)
    model_scoreB = PoissonRegressor(alpha=0.001, max_iter=2000)
    model_scoreA.fit(X_train[features], X_train["ScoreA"])
    model_scoreB.fit(X_train[features], X_train["ScoreB"])
    logging.info("Modely úspěšně natrénovány.")
except Exception as e: logging.error(f"Chyba při trénování modelů: {e}"); exit()

# --- INITIAL RATINGS, GROUPS, MATCHES - NO CHANGES ---
team_ratings = {"Germany": 2059, "Scotland": 1773, "Hungary": 1709, "Switzerland": 1847, "Spain": 2044, "Croatia": 1872, "Italy": 1960, "Albania": 1749, "Slovenia": 1695, "Denmark": 1852, "Serbia": 1827, "England": 2088, "Poland": 1832, "Netherlands": 1975, "Austria": 1813, "France": 2063, "Romania": 1616, "Ukraine": 1813, "Belgium": 1975, "Slovakia": 1700, "Georgia": 1641, "Czechia": 1749, "Turkey": 1827, "Portugal": 2009}
groups = {"A": ["Germany", "Scotland", "Hungary", "Switzerland"], "B": ["Spain", "Croatia", "Italy", "Albania"], "C": ["Slovenia", "Denmark", "Serbia", "England"], "D": ["Poland", "Netherlands", "Austria", "France"], "E": ["Belgium", "Slovakia", "Romania", "Ukraine"], "F": ["Turkey", "Georgia", "Portugal", "Czechia"]}
group_matches = [ ["Germany", "Scotland", "European Championship", "Germany"], ["Hungary", "Switzerland", "European Championship", "Germany"], ["Spain", "Croatia", "European Championship", "Germany"], ["Italy", "Albania", "European Championship", "Germany"], ["Slovenia", "Denmark", "European Championship", "Germany"], ["Serbia", "England", "European Championship", "Germany"], ["Poland", "Netherlands", "European Championship", "Germany"], ["Austria", "France", "European Championship", "Germany"], ["Romania", "Ukraine", "European Championship", "Germany"], ["Belgium", "Slovakia", "European Championship", "Germany"], ["Turkey", "Georgia", "European Championship", "Germany"], ["Portugal", "Czechia", "European Championship", "Germany"], ["Germany", "Hungary", "European Championship", "Germany"], ["Scotland", "Switzerland", "European Championship", "Germany"], ["Croatia", "Albania", "European Championship", "Germany"], ["Spain", "Italy", "European Championship", "Germany"], ["Slovenia", "Serbia", "European Championship", "Germany"], ["Denmark", "England", "European Championship", "Germany"], ["Poland", "Austria", "European Championship", "Germany"], ["Netherlands", "France", "European Championship", "Germany"], ["Slovakia", "Ukraine", "European Championship", "Germany"], ["Belgium", "Romania", "European Championship", "Germany"], ["Georgia", "Czechia", "European Championship", "Germany"], ["Turkey", "Portugal", "European Championship", "Germany"], ["Switzerland", "Germany", "European Championship", "Germany"], ["Scotland", "Hungary", "European Championship", "Germany"], ["Albania", "Spain", "European Championship", "Germany"], ["Croatia", "Italy", "European Championship", "Germany"], ["England", "Slovenia", "European Championship", "Germany"], ["Denmark", "Serbia", "European Championship", "Germany"], ["Netherlands", "Austria", "European Championship", "Germany"], ["France", "Poland", "European Championship", "Germany"], ["Slovakia", "Romania", "European Championship", "Germany"], ["Ukraine", "Belgium", "European Championship", "Germany"], ["Georgia", "Portugal", "European Championship", "Germany"], ["Czechia", "Turkey", "European Championship", "Germany"] ]

# ==========================================
# >>> simulate_match - MINIMAL CHANGE <<<
# ==========================================
def simulate_match(team_a, team_b, location, tournament, ratings, model_a, model_b, knockout=False):
    # --- NO CHANGES TO FEATURE PREP / PREDICTION / KNOCKOUT LOGIC ---
    match_features = { "PrevRatingA": [ratings[team_a]], "PrevRatingB": [ratings[team_b]], "HomeAdvantage": [(location == team_a)], "NeutralGround": [(location != team_a) and (location != team_b)], "IsFriendly": [0], "IsQualifier": [0], "IsMainTournament": [1] }
    X = pd.DataFrame(match_features)
    lambda_a = max(0.01, model_a.predict(X[features])[0])
    lambda_b = max(0.01, model_b.predict(X[features])[0])
    score_a = np.random.poisson(lambda_a)
    score_b = np.random.poisson(lambda_b)
    if knockout and score_a == score_b:
        if np.random.rand() > 0.5: score_a += 1
        else: score_b += 1

    # --- NO CHANGES TO ELO CALCULATION CALL ---
    weight = MAIN_TOURNAMENT_WEIGHT if "Final" in tournament else BASE_TOURNAMENT_WEIGHT
    delta_a, delta_b, new_a, new_b = update_elo( ratings[team_a], ratings[team_b], score_a, score_b, is_home_a=(location == team_a), is_neutral=((location != team_a) and (location != team_b)), tournament_weight=weight )

    # Update ratings dict (passed by reference)
    ratings[team_a] = new_a
    ratings[team_b] = new_b

    # *** MODIFIED: Return scores AND the new ratings ***
    # Although ratings dict is modified in place, returning helps capture specific values post-match easily.
    return score_a, score_b, new_a, new_b

# ==========================================
# >>> calculate_group_standings - NO CHANGE <<<
# ==========================================
def calculate_group_standings(matches, group_teams):
    # --- NO CHANGES HERE ---
    standings = {team: {"Points": 0, "GF": 0, "GA": 0, "GD": 0} for team in group_teams}
    for score_a, score_b, team_a, team_b in matches:
        if team_a not in standings or team_b not in standings: continue
        standings[team_a]["GF"] += score_a; standings[team_a]["GA"] += score_b; standings[team_b]["GF"] += score_b; standings[team_b]["GA"] += score_a
        if score_a > score_b: standings[team_a]["Points"] += 3
        elif score_a < score_b: standings[team_b]["Points"] += 3
        else: standings[team_a]["Points"] += 1; standings[team_b]["Points"] += 1
    for team in group_teams:
        if team in standings: standings[team]["GD"] = standings[team]["GF"] - standings[team]["GA"]
    return sorted(standings.items(), key=lambda x: (-x[1]["Points"], -x[1]["GD"], -x[1]["GF"]))

# ==========================================
# >>> run_simulation - MODIFIED FOR PER-MATCH ELO TRACKING <<<
# ==========================================
def run_simulation(n_simulations=10000): # Use lower value for testing
    # Aggregate results
    winners = {team: 0 for team in team_ratings}
    finalists = {team: 0 for team in team_ratings}
    semifinalists = {team: 0 for team in team_ratings}
    quarterfinalists = {team: 0 for team in team_ratings}

    first_simulation_details = None # Store details of sim == 0

    for sim in tqdm(range(n_simulations), desc="Simulace turnaje"):
        ratings = team_ratings.copy()
        group_results = [] # Store (score_a, score_b, team_a, team_b) for current sim

        # === Initialize structures ONLY for sim == 0 ===
        match_details = [] if sim == 0 else None            # Text descriptions of matches
        knockout_results = [] if sim == 0 else None         # Text descriptions of KO matches
        group_standings_dict = {} if sim == 0 else None     # Group standings for sim 0
        per_match_elo_snapshots = [] if sim == 0 else None  # *** NEW: List for ELO snapshots ***
        match_counter = 0 if sim == 0 else -1               # *** NEW: Match counter for ordering ***
        # current_elo_tracking = {} if sim == 0 else None   # *** REMOVED: Replaced by per_match_elo_snapshots ***

        # Store Initial Ratings snapshot for sim == 0
        if sim == 0:
             per_match_elo_snapshots.append({
                 "match_order": match_counter, # 0
                 "stage": "Initial",
                 "match_description": "Initial Ratings",
                 "team_elos": ratings.copy() # Snapshot of all initial ratings
             })

        # ===== GROUP STAGE =====
        # Using your original structure (no separate rounds stored here)
        for team_a, team_b, tournament, location in group_matches:
            # *** MODIFIED: Capture returned Elos ***
            score_a, score_b, elo_a_after, elo_b_after = simulate_match(
                team_a, team_b, location, tournament, ratings, model_scoreA, model_scoreB
            )
            group_results.append((score_a, score_b, team_a, team_b))

            if sim == 0:
                match_counter += 1
                match_desc_text = f"{team_a} {score_a}-{score_b} {team_b} (Elo: {elo_a_after:.0f} vs {elo_b_after:.0f})" # Use captured Elo
                match_details.append(match_desc_text) # Add text description

                # *** NEW: Add ELO snapshot for this match ***
                per_match_elo_snapshots.append({
                    "match_order": match_counter,
                    "stage": "Group",
                    "match_description": f"{team_a} vs {team_b}", # Keep description simple
                    "team_elos": {team_a: elo_a_after, team_b: elo_b_after} # Store Elo for involved teams
                })

        # ===== CALCULATE GROUP STANDINGS & QUALIFIERS - NO CHANGE TO LOGIC =====
        qualified = []
        third_place = []
        for group_name, teams_in_group in groups.items():
            group_games = [(s_a, s_b, t_a, t_b) for s_a, s_b, t_a, t_b in group_results
                          if t_a in teams_in_group and t_b in teams_in_group]
            standings = calculate_group_standings(group_games, teams_in_group)
            if sim == 0: group_standings_dict[group_name] = standings # Store standings for sim 0
            if standings:
                qualified.append(standings[0][0]) # 1st
                if len(standings) > 1: qualified.append(standings[1][0]) # 2nd
                if len(standings) > 2: third_place.append((standings[2][0], standings[2][1], group_name)) # 3rd
        third_place.sort(key=lambda x: (-x[1]["Points"], -x[1]["GD"], -x[1]["GF"]))
        best_third = [x[0] for x in third_place[:4]]
        qualified.extend(best_third)

        # Validation - Check if 16 teams qualified
        if len(qualified) != 16:
            logging.warning(f"Sim {sim}: Incorrect number of qualified teams ({len(qualified)}). Skipping KO stage.")
            if sim == 0:
                first_simulation_details = { "error": f"Kvalifikováno pouze {len(qualified)} týmů.", "per_match_elo_snapshots": per_match_elo_snapshots }
            continue

        # Prepare details dict for sim == 0 before KO stage
        if sim == 0:
            third_place_details_for_dict = [(x[0], x[2], x[1]["Points"], x[1]["GD"], x[1]["GF"]) for x in third_place]
            # *** MODIFIED: Structure of first_simulation_details ***
            first_simulation_details = {
                "match_results": match_details, # List of text results from group stage
                "group_standings": group_standings_dict,
                "third_place_teams": third_place_details_for_dict,
                "qualified_teams": qualified[:], # Store a COPY
                "knockout_results": knockout_results, # Empty list, filled below
                "final_ratings": None, # Filled after final
                "per_match_elo_snapshots": per_match_elo_snapshots # List of ELO snapshots so far
                # REMOVED: "ratings_after_group", "elo_tracking"
            }

        # ===== KNOCKOUT STAGES - PRESERVING YOUR ORIGINAL LOGIC STRUCTURE =====
        try:
            # --- Osmifinále - Your original pairing logic ---
            r16_matches = [ # Using your exact pairings from the provided code
                [qualified[0], qualified[15]],   # 1A vs 3D/E/F
                [qualified[1], qualified[2]],    # 1B vs 3A/D/E/F -> ?? Should be 3rd place? Check UEFA rules. Assuming this is placeholder.
                [qualified[4], qualified[13]],   # 1C vs 3D/E/F
                [qualified[5], qualified[12]],   # 1D vs 2F -> ?? Should be 3rd place?
                [qualified[6], qualified[11]],   # 1E vs 3A/B/C
                [qualified[7], qualified[10]],   # 1F vs 3A/B/C
                [qualified[8], qualified[9]],    # 2A vs 2B
                [qualified[3], qualified[14]]    # 2D vs 2E
            ]
            # --- Simulace osmifinále - Your original loop ---
            r16_winners = []
            stage_name = "R16" # Define stage name
            for i, (a, b) in enumerate(r16_matches):
                # *** Call simulate_match WITHOUT capturing return Elos (as per your original code) ***
                s_a, s_b, _, _ = simulate_match(a, b, "Germany", f"European Championship {stage_name}", ratings, model_scoreA, model_scoreB, knockout=True)
                winner = a if s_a > s_b else b
                r16_winners.append(winner)
                quarterfinalists[a] += 1 # Aggregate counter update
                quarterfinalists[b] += 1

                if sim == 0:
                    match_counter += 1
                    # *** Read updated Elo values AFTER the match ***
                    elo_a_now = ratings[a]
                    elo_b_now = ratings[b]
                    # Add text description including the read Elo values
                    knockout_results.append(
                        f"{stage_name}: {a} {s_a}-{s_b} {b} (Winner: {winner}, Elo: {elo_a_now:.0f} vs {elo_b_now:.0f})"
                    )
                    # *** NEW: Add ELO snapshot using the read values ***
                    first_simulation_details["per_match_elo_snapshots"].append({
                        "match_order": match_counter,
                        "stage": stage_name,
                        "match_description": f"{a} vs {b}",
                        "team_elos": {a: elo_a_now, b: elo_b_now}
                    })

            # --- Čtvrtfinále - Your original pairing logic ---
            qf_matches = [
                [r16_winners[0], r16_winners[1]],
                [r16_winners[2], r16_winners[3]],
                [r16_winners[4], r16_winners[5]],
                [r16_winners[6], r16_winners[7]]
            ]
            # --- Simulace čtvrtfinále - Your original loop ---
            qf_winners = []
            stage_name = "QF" # Define stage name
            for a, b in qf_matches:
                 # *** Call simulate_match WITHOUT capturing return Elos ***
                s_a, s_b, _, _ = simulate_match(a, b, "Germany", f"European Championship {stage_name}", ratings, model_scoreA, model_scoreB, knockout=True)
                winner = a if s_a > s_b else b
                qf_winners.append(winner)
                semifinalists[a] += 1 # Aggregate counter update
                semifinalists[b] += 1

                if sim == 0:
                    match_counter += 1
                    # *** Read updated Elo values AFTER the match ***
                    elo_a_now = ratings[a]
                    elo_b_now = ratings[b]
                    # Add text description including the read Elo values
                    first_simulation_details["knockout_results"].append(
                         f"{stage_name}: {a} {s_a}-{s_b} {b} (Winner: {winner}, Elo: {elo_a_now:.0f} vs {elo_b_now:.0f})"
                    )
                    # *** NEW: Add ELO snapshot using the read values ***
                    first_simulation_details["per_match_elo_snapshots"].append({
                        "match_order": match_counter,
                        "stage": stage_name,
                        "match_description": f"{a} vs {b}",
                        "team_elos": {a: elo_a_now, b: elo_b_now}
                    })

            # --- Semifinále - Your original pairing logic ---
            sf_matches = [
                [qf_winners[0], qf_winners[1]],
                [qf_winners[2], qf_winners[3]]
            ]
            # --- Simulace semifinále - Your original loop ---
            final_teams = []
            stage_name = "SF" # Define stage name
            for a, b in sf_matches:
                 # *** Call simulate_match WITHOUT capturing return Elos ***
                s_a, s_b, _, _ = simulate_match(a, b, "Germany", f"European Championship {stage_name}", ratings, model_scoreA, model_scoreB, knockout=True)
                winner = a if s_a > s_b else b
                final_teams.append(winner)
                finalists[a] += 1 # Aggregate counter update
                finalists[b] += 1

                if sim == 0:
                    match_counter += 1
                    # *** Read updated Elo values AFTER the match ***
                    elo_a_now = ratings[a]
                    elo_b_now = ratings[b]
                    # Add text description including the read Elo values
                    first_simulation_details["knockout_results"].append(
                        f"{stage_name}: {a} {s_a}-{s_b} {b} (Winner: {winner}, Elo: {elo_a_now:.0f} vs {elo_b_now:.0f})"
                    )
                    # *** NEW: Add ELO snapshot using the read values ***
                    first_simulation_details["per_match_elo_snapshots"].append({
                        "match_order": match_counter,
                        "stage": stage_name,
                        "match_description": f"{a} vs {b}",
                        "team_elos": {a: elo_a_now, b: elo_b_now}
                    })

            # --- Finále - Your original logic ---
            team_a = final_teams[0]
            team_b = final_teams[1]
            stage_name = "Final" # Define stage name
             # *** Call simulate_match WITHOUT capturing return Elos ***
            s_a, s_b, _, _ = simulate_match(team_a, team_b, "Germany", f"European Championship {stage_name}", ratings, model_scoreA, model_scoreB, knockout=True)
            winner = team_a if s_a > s_b else team_b
            winners[winner] += 1 # Aggregate counter update

            if sim == 0:
                match_counter += 1
                # *** Read updated Elo values AFTER the match ***
                elo_a_now = ratings[team_a]
                elo_b_now = ratings[team_b]
                # Add text description including the read Elo values
                first_simulation_details["knockout_results"].append(
                     f"{stage_name}: {team_a} {s_a}-{s_b} {team_b} (Winner: {winner}, Elo: {elo_a_now:.0f} vs {elo_b_now:.0f})"
                )
                # *** NEW: Add ELO snapshot using the read values ***
                first_simulation_details["per_match_elo_snapshots"].append({
                    "match_order": match_counter,
                    "stage": stage_name,
                    "match_description": f"{team_a} vs {team_b}",
                    "team_elos": {team_a: elo_a_now, team_b: elo_b_now}
                })
                # Store final ratings dict
                first_simulation_details["final_ratings"] = ratings.copy()


        except (IndexError, ValueError, KeyError) as e: # Your original error handling
            logging.error(f"Sim {sim}: Chyba při zpracování KO fáze: {e}")
            # Ensure error is recorded for sim 0
            if sim == 0 and first_simulation_details and "error" not in first_simulation_details:
                first_simulation_details["error"] = f"Chyba KO fáze: {e}"
            continue
        except Exception as e: # Your original error handling
            logging.error(f"Sim {sim}: Neočekávaná chyba KO fáze: {e}")
            logging.exception("Detail:")
            if sim == 0 and first_simulation_details and "error" not in first_simulation_details:
                first_simulation_details["error"] = f"Chyba KO fáze: {e}"
            continue

    # ===== END OF SIMULATION LOOP =====

    # --- Calculate probabilities - NO CHANGE ---
    total_sims_run = n_simulations
    win_probs = {t: (c / total_sims_run) * 100 for t, c in winners.items()}
    final_probs = {t: (c / total_sims_run) * 100 for t, c in finalists.items()}
    semi_probs = {t: (c / total_sims_run) * 100 for t, c in semifinalists.items()}
    quarter_probs = {t: (c / total_sims_run) * 100 for t, c in quarterfinalists.items()}

    return win_probs, final_probs, semi_probs, quarter_probs, first_simulation_details

# ==============================================================================
# >>> DATABASE FUNCTIONS - NO CHANGES HERE <<<
# Using the versions provided/confirmed previously.
# save_first_run_details_to_db *MUST* be the version that processes
# 'per_match_elo_snapshots'
# ==============================================================================
def create_connection(db_file):
    # --- NO CHANGE ---
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"DB: Úspěšně připojeno k {db_file}")
    except sqlite3.Error as e: logging.error(f"DB: Chyba při připojování: {e}")
    return conn

def save_probabilities_to_db(conn, win_p, final_p, semi_p, quarter_p):
     # --- NO CHANGE ---
    if conn is None: logging.error("DB: Nelze uložit pravděpodobnosti: Chybí připojení."); return
    try:
        table_name = 'simulation_probabilities'; logging.info(f"DB: Příprava pravděpodobností pro '{table_name}'...")
        all_teams = list(team_ratings.keys())
        results_list = [{'nationality': team, 'win_prob': win_p.get(team, 0), 'final_prob': final_p.get(team, 0), 'semi_prob': semi_p.get(team, 0), 'quarter_prob': quarter_p.get(team, 0)} for team in all_teams]
        results_df = pd.DataFrame(results_list)
        logging.info(f"DB: Ukládání pravděpodobností do '{table_name}'..."); results_df.to_sql(table_name, conn, if_exists='replace', index=False)
        logging.info(f"DB: Tabulka '{table_name}' naplněna.")
    except Exception as e: logging.error(f"DB: Chyba při ukládání pravděpodobností: {e}"); logging.exception("Detail:")

def save_first_run_details_to_db(conn, details_dict, simulation_id=1):
    # ... (checks and variable initializations) ...
    if conn is None: logging.error("DB: Cannot save details: Connection is missing."); return
    if not details_dict or not isinstance(details_dict, dict): logging.warning(f"DB: Details data missing or not a dictionary: {details_dict}"); return
    details_table_name = 'simulation_run_details'; elo_table_name = 'elo_snapshots'
    details_to_insert = []; elo_snapshots_to_insert = []

    try:
        # --- Prepare simulation_run_details data ---
        # ... (Add match_results descriptions - NO CHANGE) ...
        if "match_results" in details_dict and isinstance(details_dict["match_results"], list): details_to_insert.extend([(simulation_id, 'Group Match', str(desc)) for desc in details_dict["match_results"]])

        # *** MODIFY HOW STANDINGS STRING IS CREATED ***
        if "group_standings" in details_dict and isinstance(details_dict["group_standings"], dict):
            for group_name, standings in details_dict["group_standings"].items():
                try:
                    # Create the string parts for each team including GF and GA
                    team_standing_parts = []
                    for i, (team, stats) in enumerate(standings, 1):
                        if isinstance(stats, dict):
                            # Format: "1.Team (Ptsb,GD,GF-GA)"
                            team_part = f"{i}.{team.strip()} ({stats.get('Points',0)}b,{stats.get('GD',0):+},{stats.get('GF',0)}-{stats.get('GA',0)})"
                            team_standing_parts.append(team_part)

                    # Join the parts into the final description string
                    standing_str = ", ".join(team_standing_parts)
                    details_to_insert.append((simulation_id, f"{group_name} Standing", standing_str))
                    logging.info(f"DB: Prepared standing string for {group_name}: {standing_str}") # Log the created string
                except Exception as e:
                    logging.warning(f"DB: Could not process standings for group {group_name}: {e}")
        # *** END OF STANDINGS STRING MODIFICATION ***

        # ... (Add third_place_teams, qualified_teams, knockout_results, error - NO CHANGE) ...
        if "third_place_teams" in details_dict and isinstance(details_dict["third_place_teams"], list):
            try: details_to_insert.append((simulation_id, '3rd Place Ranking', ", ".join([f"{i}.{t[0]}({t[1]},{t[2]}b,{t[3]:+})" for i, t in enumerate(details_dict["third_place_teams"], 1) if len(t) >= 4])))
            except Exception as e: logging.warning(f"DB: Could not process 3rd place ranking: {e}")
        if "qualified_teams" in details_dict and isinstance(details_dict["qualified_teams"], list) and len(details_dict["qualified_teams"]) == 16:
            details_to_insert.append((simulation_id, 'Best 3rd Qualifiers', ", ".join(details_dict["qualified_teams"][12:])))
            try: details_to_insert.append((simulation_id, 'Qualified R16 List', json.dumps(details_dict["qualified_teams"])))
            except Exception as e: logging.error(f"DB: Error saving qualified teams JSON: {e}")
        if "knockout_results" in details_dict and isinstance(details_dict["knockout_results"], list):
             for desc in details_dict["knockout_results"]:
                stage="Knockout"; desc_str = str(desc)
                if desc_str.startswith("R16:"): stage = "R16"
                elif desc_str.startswith("QF:"): stage = "QF"
                elif desc_str.startswith("SF:"): stage = "SF"
                elif desc_str.startswith("Final:"): stage = "Final"
                details_to_insert.append((simulation_id, stage, desc_str))
        if "error" in details_dict: details_to_insert.append((simulation_id, "Simulation Error", str(details_dict["error"])))


        # --- Prepare elo_snapshots data (No change needed here) ---
        if "per_match_elo_snapshots" in details_dict and isinstance(details_dict["per_match_elo_snapshots"], list):
            # ... (existing logic for elo_snapshots_to_insert) ...
             for snapshot in details_dict["per_match_elo_snapshots"]:
                if not isinstance(snapshot, dict): continue
                match_order = snapshot.get("match_order", -1); stage = snapshot.get("stage", "Unknown"); match_desc = snapshot.get("match_description", "N/A"); team_elos = snapshot.get("team_elos", {})
                if not isinstance(team_elos, dict): continue
                for team, elo in team_elos.items():
                    if elo is not None: elo_snapshots_to_insert.append((simulation_id, match_order, stage, match_desc, team, elo))
        else: logging.warning("DB: Key 'per_match_elo_snapshots' not found. No per-match ELO data saved.")

        # --- Database Operations (No change needed here) ---
        cursor = conn.cursor()
        logging.info(f"DB: Deleting old records for simulation_id={simulation_id}..."); cursor.execute(f"DELETE FROM {details_table_name} WHERE simulation_id = ?", (simulation_id,)); cursor.execute(f"DELETE FROM {elo_table_name} WHERE simulation_id = ?", (simulation_id,))
        logging.info(f"DB: Inserting {len(details_to_insert)} records into '{details_table_name}'...");
        if details_to_insert: cursor.executemany(f"INSERT INTO {details_table_name} (simulation_id, stage, description) VALUES (?, ?, ?)", details_to_insert)
        logging.info(f"DB: Inserting {len(elo_snapshots_to_insert)} records into '{elo_table_name}'...");
        if elo_snapshots_to_insert: cursor.executemany(f"INSERT INTO {elo_table_name} (simulation_id, match_order, stage, match_description, nationality, elo_after_match) VALUES (?, ?, ?, ?, ?, ?)", elo_snapshots_to_insert)
        conn.commit(); logging.info(f"DB: Tables '{details_table_name}' and '{elo_table_name}' populated successfully.")

    except Exception as e:
        logging.error(f"DB: General error while saving details for simulation_id={simulation_id}: {e}")
        conn.rollback()

# ==============================================================================
# >>> MAIN EXECUTION BLOCK - NO CHANGE <<<
# ==============================================================================
if __name__ == "__main__":
    # --- NO CHANGES HERE (uses the functions above) ---
    print("Probíhá simulace turnaje Euro 2024...")
    try:
        win_probs, final_probs, semi_probs, quarter_probs, first_run_details_dict = run_simulation(n_simulations=10000) # Test value
        simulation_successful = True
        if first_run_details_dict is None: logging.error("Nebyla vrácena data z první simulace."); simulation_successful = False; win_probs = {}
        elif "error" in first_run_details_dict: logging.warning(f"První simulace zaznamenala chybu: {first_run_details_dict['error']}")
    except Exception as e:
        logging.error(f"Chyba během běhu run_simulation: {e}"); simulation_successful = False; win_probs, first_run_details_dict = {}, None

    if simulation_successful and win_probs:
        conn = create_connection(DATABASE_PATH)
        if conn:
            try:
                save_probabilities_to_db(conn, win_probs, final_probs, semi_probs, quarter_probs)
                if first_run_details_dict is not None: save_first_run_details_to_db(conn, first_run_details_dict)
                else: logging.warning("Detaily první simulace nebyly dostupné pro uložení.")
            finally: conn.close()
            print("\nVýsledky byly uloženy do databáze data/database.db")
        else: logging.error("DB: Nepodařilo se připojit k DB pro uložení.")
    elif not simulation_successful: print("\nVýsledky se nepodařilo uložit kvůli chybě v simulaci.")
    else: print("\nSimulace proběhla, ale nebyly vypočteny pravděpodobnosti pro uložení.")

    print("\n=== VÝSLEDKY SIMULACE (SHRNUTÍ PRAVDĚPODOBNOSTÍ) ===\n")
    if win_probs:
        print("Top 10 pravděpodobností vítězství:")
        for team, prob in sorted(win_probs.items(), key=lambda x: -x[1])[:10]: print(f"{team}: {prob:.1f}%")
    else: print("Nebylo možné zobrazit pravděpodobnosti.")

    print("\n=== DETAILY PRVNÍ SIMULACE ===")
    if first_run_details_dict and 'error' not in first_run_details_dict:
        if "match_results" in first_run_details_dict: print("\nVýsledky zápasů ve skupinách:"); [print(r) for r in first_run_details_dict["match_results"]]
        if "group_standings" in first_run_details_dict:
             print("\nVýsledky skupin:");
             for group_name, standings in first_run_details_dict["group_standings"].items():
                 print(f"\n Skupina {group_name}:"); [print(f" {i}. {t}: {s.get('Points', 0)}b (GD:{s.get('GD', 0)})") for i, (t, s) in enumerate(standings, 1)]
        if "third_place_teams" in first_run_details_dict: print("\nPořadí týmů na 3. místech:"); print(" ".join([f"{i}.{t[0]}({t[1]},{t[2]}b,{t[3]:+})" for i, t in enumerate(first_run_details_dict["third_place_teams"], 1) if len(t) >= 4]))
        if "qualified_teams" in first_run_details_dict: print("\nPostupující týmy:"); print("1-2:", ", ".join(first_run_details_dict["qualified_teams"][:12])); print("3.:", ", ".join(first_run_details_dict["qualified_teams"][12:16]))
        if "knockout_results" in first_run_details_dict: print("\nVýsledky vyřazovacích kol:"); [print(r) for r in first_run_details_dict["knockout_results"]]
        if "final_ratings" in first_run_details_dict: print("\nKonečné ELO:"); [print(f" {t}: {r:.0f}") for t, r in sorted(first_run_details_dict["final_ratings"].items(), key=lambda x: -x[1])]
    elif first_run_details_dict and 'error' in first_run_details_dict: print(f"\nPrvní simulace skončila s chybou: {first_run_details_dict['error']}")
    else: print("\nNebylo možné zobrazit detaily první simulace.")