# app/routes.py
# *** ENSURE 'request' IS INCLUDED IN THIS IMPORT FROM FLASK ***
from flask import render_template, current_app as app, g, abort, url_for, request, make_response
import sqlite3
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
import logging
import os
import re
import numpy as np
from io import StringIO
import urllib.parse

# --- (Logging, DB Path, Formations, DB Helpers - NO CHANGES) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE = os.path.join(BASE_DIR, 'data', 'database.db')

formations = {
    # ... (keep your formations dictionary here) ...
    '4-4-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Left Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Centre Forward', 'Centre Forward'],
    '4-3-3': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Centre Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Left Wing Forward', 'Right Wing Forward', 'Centre Forward'],
    '3-5-2': ['Goalkeeper', 'Centre Back', 'Centre Back', 'Centre Back', 'Left Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Centre Forward', 'Centre Forward'],
    '4-2-3-1': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Defensive Midfielder', 'Defensive Midfielder', 'Attacking Midfielder', 'Left Midfielder', 'Right Midfielder', 'Centre Forward'],
    '4-1-4-1': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Defensive Midfielder', 'Left Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Centre Forward'],
    '5-3-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Centre Back', 'Right Back', 'Centre Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Second Striker', 'Centre Forward'],
    '3-4-3': ['Goalkeeper', 'Centre Back', 'Centre Back', 'Centre Back', 'Right Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Left Midfielder', 'Right Wing Forward', 'Centre Forward', 'Left Wing Forward'],
    '4-4-1-1': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Left Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Attacking Midfielder', 'Centre Forward'],
    '4-3-1-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Left Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Attacking Midfielder', 'Centre Forward', 'Centre Forward'],
    '4-2-2-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Defensive Midfielder', 'Defensive Midfielder', 'Attacking Midfielder', 'Attacking Midfielder', 'Centre Forward', 'Centre Forward'],
    '4-3-2-1': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Left Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Attacking Midfielder', 'Attacking Midfielder', 'Centre Forward'],
    '3-5-1-1': ['Goalkeeper', 'Centre Back', 'Centre Back', 'Centre Back', 'Left Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Second Striker', 'Centre Forward'],
    '4-1-3-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Defensive Midfielder', 'Left Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Second Striker', 'Centre Forward'],
    '4-4-2 Diamond': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Left Midfielder', 'Right Midfielder', 'Defensive Midfielder', 'Attacking Midfielder', 'Centre Forward', 'Centre Forward'],
    '5-2-1-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Centre Back', 'Right Back', 'Defensive Midfielder', 'Defensive Midfielder', 'Attacking Midfielder', 'Centre Forward', 'Centre Forward'],
    '4-3-3 (Attack)': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Centre Midfielder', 'Centre Midfielder', 'Attacking Midfielder', 'Left Wing Forward', 'Right Wing Forward', 'Centre Forward'],
    '4-1-2-1-2': ['Goalkeeper', 'Left Back', 'Centre Back', 'Centre Back', 'Right Back', 'Defensive Midfielder', 'Left Midfielder', 'Right Midfielder', 'Attacking Midfielder', 'Centre Forward', 'Centre Forward'],
    '3-4-2-1': ['Goalkeeper', 'Centre Back', 'Centre Back', 'Centre Back', 'Left Midfielder', 'Centre Midfielder', 'Centre Midfielder', 'Right Midfielder', 'Attacking Midfielder', 'Attacking Midfielder', 'Centre Forward']
}
groups = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Slovenia", "Denmark", "Serbia", "England"],
    "D": ["Poland", "Netherlands", "Austria", "France"],
    "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "F": ["Turkey", "Georgia", "Portugal", "Czechia"]
}
COUNTRY_CODES = {
    "Germany": "DE", "Scotland": "GB-SCT", "Hungary": "HU", "Switzerland": "CH", # Note: GB-SCT for Scotland emoji usually needs OS support
    "Spain": "ES", "Croatia": "HR", "Italy": "IT", "Albania": "AL",
    "Slovenia": "SI", "Denmark": "DK", "Serbia": "RS", "England": "GB-ENG", # Note: GB-ENG for England
    "Poland": "PL", "Netherlands": "NL", "Austria": "AT", "France": "FR",
    "Belgium": "BE", "Slovakia": "SK", "Romania": "RO", "Ukraine": "UA",
    "Türkiye": "TR", "Georgia": "GE", "Portugal": "PT", "Czechia": "CZ"
    # Add others as needed
}
def get_flag_emoji(country_name):
    code = COUNTRY_CODES.get(country_name)
    if not code or len(code) != 2: # Basic check, needs refinement for GB-* codes
         # Fallback for compound codes or missing codes
         if country_name == "England": return "🏴󠁧󠁢󠁥󠁮󠁧󠁿"
         if country_name == "Scotland": return "🏴󠁧󠁢󠁳󠁣󠁴󠁿"
         # Wales example: return "🏴󠁧󠁢󠁷󠁬󠁳󠁿"
         return "🏳️" # Default white flag
    # Formula to convert 2-letter code to emoji
    return chr(ord('🇦') + ord(code[0]) - ord('A')) + chr(ord('🇦') + ord(code[1]) - ord('A'))

# Inside your index() route function:

def get_database_path():
    # --- (No changes here) ---
    if 'BASE_DIR' not in app.config:
        app.config['BASE_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(app.config['BASE_DIR'], 'data', 'database.db')

def get_db():
    # --- (No changes here) ---
    if 'db' not in g:
        db_path = get_database_path()
        try:
            g.db = sqlite3.connect(db_path)
            g.db.row_factory = sqlite3.Row
            logging.info(f"DB connection successful to {db_path}")
        except sqlite3.Error as e: logging.error(f"DB connection error: {e}"); abort(500)
    return g.db

def _get_overview_stats(conn, scope):
    """Generates the overview statistics."""
    team_stats = []
    page_title_suffix = ""
    numeric_cols_desc = ['age', 'height', 'weight', 'overall_rating']
    df_desc = None
    query = ""

    if scope == 'squad':
        query = """ SELECT p.nationality, p.age, p.height, p.weight, p.overall_rating FROM players p JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality """
        page_title_suffix = " (Soupiska)"
    else:
        scope = 'all' # Ensure default
        query = f"SELECT nationality, {', '.join(numeric_cols_desc)} FROM players"
        page_title_suffix = " (Všichni hráči)"

    try:
        df_desc = pd.read_sql_query(query, conn)
        if df_desc is not None and not df_desc.empty:
            for col in numeric_cols_desc: df_desc[col] = pd.to_numeric(df_desc[col], errors='coerce')
            df_desc.dropna(subset=numeric_cols_desc, how='all', inplace=True)
            if not df_desc.empty:
                grouped_stats = df_desc.groupby('nationality')[numeric_cols_desc].agg(
                    avg_age=('age', 'mean'), avg_height=('height', 'mean'),
                    avg_weight=('weight', 'mean'), avg_rating=('overall_rating', 'mean'),
                    player_count=('overall_rating', 'size')
                ).reset_index()
                for col in ['avg_age', 'avg_height', 'avg_weight', 'avg_rating']:
                    grouped_stats[col] = grouped_stats[col].round(1)
                team_stats = grouped_stats.to_dict('records')
                logging.info(f"Overview helper calculated stats for {len(team_stats)} teams.")
            else:
                logging.warning(f"Overview helper: No valid numeric data for scope '{scope}'.")
        else:
             logging.warning(f"Overview helper: No data fetched for scope '{scope}'.")
    except Exception as e:
        logging.error(f"Error in _get_overview_stats: {e}")
        team_stats = [] # Return empty list on error

    return team_stats, page_title_suffix

def _get_two_team_comparison_data(conn, team1, team2, scope):
    """Generates the two-team comparison data."""
    team_comparison_data = None
    attributes_to_compare = [
        'overall_rating', 'offensive_awareness', 'ball_control', 'dribbling', 'tight_possession',
        'low_pass', 'lofted_pass', 'finishing', 'heading', 'set_piece_taking', 'curl',
        'defensive_awareness', 'tackling', 'aggression', 'defensive_engagement',
        'speed', 'acceleration', 'kicking_power', 'jumping', 'physical_contact',
        'balance', 'stamina', 'gk_awareness', 'gk_catching', 'gk_parrying',
        'gk_reflexes', 'gk_reach']
    calculated_averages = {}
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(players)")
        actual_player_cols = {info[1] for info in cursor.fetchall()}
        cols_to_select_comp = [attr for attr in attributes_to_compare if attr in actual_player_cols]
        cols_str_comp_quoted = ', '.join([f'p."{col}"' for col in cols_to_select_comp])

        for team in [team1, team2]:
            df_team = None
            query_base_comp = f"SELECT {cols_str_comp_quoted} FROM players p "
            if scope == 'squad': query_team = query_base_comp + "JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality WHERE p.nationality = ?"
            else: query_team = query_base_comp + "WHERE p.nationality = ?"
            df_team = pd.read_sql_query(query_team, conn, params=(team,))
            if df_team is not None and not df_team.empty:
                team_averages = {attr: round(pd.to_numeric(df_team[attr], errors='coerce').mean(), 1) if pd.notna(pd.to_numeric(df_team[attr], errors='coerce').mean()) else None for attr in cols_to_select_comp if attr in df_team.columns}
                calculated_averages[team] = team_averages
            else: calculated_averages[team] = {attr: None for attr in cols_to_select_comp}

        team1_avg_stats = calculated_averages.get(team1); team2_avg_stats = calculated_averages.get(team2)
        if team1_avg_stats and team2_avg_stats:
            team_comparison_data = []
            for attr in attributes_to_compare:
                val1 = team1_avg_stats.get(attr); val2 = team2_avg_stats.get(attr)
                comp_entry = {'name': attr.replace('_', ' ').title(), 't1_value': val1 if pd.notna(val1) else 'N/A', 't2_value': val2 if pd.notna(val2) else 'N/A', 't1_better': False, 't2_better': False, 'diff': 0}
                if pd.notna(val1) and pd.notna(val2):
                    comp_entry['t1_better'] = val1 > val2; comp_entry['t2_better'] = val2 > val1
                    diff = abs(round(val1 - val2, 1)); comp_entry['diff'] = int(diff) if diff == int(diff) else diff
                team_comparison_data.append(comp_entry)
    except Exception as e:
        logging.error(f"Error in _get_two_team_comparison_data: {e}")
        team_comparison_data = None # Return None on error

    return team_comparison_data

def _get_single_team_analysis(conn, team_name, scope, attributes_to_analyze, include_gk):
    """Generates the single-team min/max analysis data."""
    analysis_results = {}
    available_attributes = _get_available_attributes() # Get full list
    numeric_attrs_in_db = []

    try:
        base_cols = ['player_name', 'nationality', 'primary_position']
        cursor = conn.cursor(); cursor.execute("PRAGMA table_info(players)"); actual_cols = {info[1] for info in cursor.fetchall()}
        numeric_attrs_in_db = [attr for attr in available_attributes if attr not in base_cols and attr in actual_cols]
        # Use only valid attributes requested, or defaults if none provided/valid
        valid_attrs_to_analyze = [attr for attr in attributes_to_analyze if attr in numeric_attrs_in_db]
        if not valid_attrs_to_analyze:
             valid_attrs_to_analyze = [attr for attr in ['height', 'weight', 'age', 'overall_rating', 'speed', 'acceleration', 'finishing', 'tackling', 'dribbling', 'low_pass', 'stamina'] if attr in numeric_attrs_in_db]

        all_cols_to_select = list(dict.fromkeys(base_cols + valid_attrs_to_analyze))
        cols_str_select_quoted = ', '.join([f'p."{col}"' for col in all_cols_to_select])

        df_single_team = None
        query_base = f"SELECT {cols_str_select_quoted} FROM players p "
        if scope == 'squad': query_single = query_base + "JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality WHERE p.nationality = ?"
        else: query_single = query_base + "WHERE p.nationality = ?"
        df_single_team = pd.read_sql_query(query_single, conn, params=(team_name,))

        if df_single_team is not None and not df_single_team.empty:
            df_analysis_subset = df_single_team.copy()
            if include_gk == 'no' and 'primary_position' in df_analysis_subset.columns:
                condition = df_analysis_subset['primary_position'].astype(str).str.strip().str.lower() != 'gk'
                df_analysis_subset = df_analysis_subset.loc[condition].copy()

            if not df_analysis_subset.empty:
                for attr in valid_attrs_to_analyze: # Only iterate through attributes we intend to analyze
                    if attr in df_analysis_subset.columns:
                        df_analysis_subset[attr] = pd.to_numeric(df_analysis_subset[attr], errors='coerce')
                        valid_data = df_analysis_subset.dropna(subset=[attr])
                        if not valid_data.empty:
                            max_val, min_val = valid_data[attr].max(), valid_data[attr].min()
                            max_players, min_players = ["N/A"], ["N/A"]
                            if 'player_name' in valid_data.columns:
                                 max_players = valid_data.loc[valid_data[attr] == max_val, 'player_name'].tolist()
                                 min_players = valid_data.loc[valid_data[attr] == min_val, 'player_name'].tolist()
                            analysis_results[attr] = {
                                'name': attr.replace('_', ' ').title(),
                                'max_val': int(max_val) if pd.notna(max_val) and max_val == int(max_val) else round(max_val, 1) if pd.notna(max_val) else 'N/A',
                                'max_players': max_players,
                                'min_val': int(min_val) if pd.notna(min_val) and min_val == int(min_val) else round(min_val, 1) if pd.notna(min_val) else 'N/A',
                                'min_players': min_players }
                        else: analysis_results[attr] = {'name': attr.replace('_', ' ').title(), 'error': 'No valid data'}
                    else: analysis_results[attr] = {'name': attr.replace('_', ' ').title(), 'error': 'Attr. unavailable'}
            else: # Handle empty subset
                 for attr in valid_attrs_to_analyze: analysis_results[attr] = {'name': attr.replace('_', ' ').title(), 'error': 'No players match criteria'}
        else: # Handle no data fetched initially
             logging.warning(f"Single team helper: No data fetched for {team_name}"); analysis_results = {} # Return empty results
    except Exception as e:
        logging.error(f"Error in _get_single_team_analysis: {e}")
        analysis_results = {} # Return empty results on error

    return analysis_results

def _get_available_attributes():
    """Returns the list of attributes available for analysis/export."""
    # Consistent list used for checkboxes and potentially filtering export columns
    available_attributes = [
            'age', 'height', 'weight', 'overall_rating', 'potential',
            'offensive_awareness', 'ball_control', 'dribbling', 'tight_possession',
            'low_pass', 'lofted_pass', 'finishing', 'heading', 'set_piece_taking', 'curl',
            'defensive_awareness', 'tackling', 'aggression', 'defensive_engagement',
            'speed', 'acceleration', 'kicking_power', 'jumping', 'physical_contact',
            'balance', 'stamina', 'gk_awareness', 'gk_catching', 'gk_parrying',
            'gk_reflexes', 'gk_reach'
        ]
    available_attributes.sort()
    return available_attributes




@app.teardown_appcontext
def close_db(exception=None):
    # --- (No changes here) ---
    db = g.pop('db', None)
    if db is not None: db.close(); logging.info("DB connection closed.")

@app.route('/')
@app.route('/index')
def index():
    conn = get_db()
    teams_data = [] # Store dicts {name: '...', flag: '...'}
    try:
        cursor = conn.execute("SELECT DISTINCT nationality FROM rosters ORDER BY nationality")
        # Fetch teams and add flags
        teams_raw = [row['nationality'] for row in cursor.fetchall()]
        teams_data = [{'name': team, 'flag': get_flag_emoji(team)} for team in teams_raw]
    except Exception as e:
        logging.error(f"Error fetching teams: {e}")
    # Pass teams_data instead of just teams list
    return render_template('index.html', title='Domovská stránka', teams_data=teams_data)


@app.route('/tym/<team_name>')
def team_roster(team_name):
    # --- (No changes here) ---
    conn = get_db()
    players_list = []; formation_name = "N/A"; total_team_rating = 0
    try:
        cursor = conn.execute("SELECT player_name, assigned_position, overall_rating_in_position, formation_name FROM rosters WHERE nationality = ?", (team_name,))
        roster_data_raw = [dict(row) for row in cursor.fetchall()]
        if not roster_data_raw: abort(404)
        formation_name = roster_data_raw[0]['formation_name']
        total_team_rating = sum(p['overall_rating_in_position'] for p in roster_data_raw)
        if formation_name in formations:
            pos_map = {pos: i for i, pos in enumerate(formations[formation_name])}
            players_list = sorted(roster_data_raw, key=lambda p: pos_map.get(p['assigned_position'], 99))
        else: players_list = sorted(roster_data_raw, key=lambda p: p['player_name'])
    except Exception as e: logging.error(f"Error reading roster for {team_name}: {e}"); abort(500)
    return render_template('team_roster.html', title=f'Soupiska - {team_name}', team_name=team_name, formation_name=formation_name, players=players_list, total_team_rating=total_team_rating)

# --- (Keep simulation_results route - NO CHANGES) ---
@app.route('/simulace')
def simulation_results():
    conn = get_db()
    probabilities = []
    parsed_run_details = {
        'Group': {'Match 1': [], 'Match 2': [], 'Match 3': []},
        'R16': [], 'QF': [], 'SF': [], 'Final': []
    }
    third_place_ranking_details = []
    win_prob_chart_json = "{}"
    elo_evolution_chart_json = "{}"
    qualified_r16_teams = []
    parsed_standings = {} # Will be populated in the first pass
    stacked_prob_chart_json = "{}"
    elo_comparison_chart_json = "{}"
    group_definitions = groups

    try:
        # 1. Fetch Probabilities (No Change)
        cursor_prob = conn.execute("SELECT * FROM simulation_probabilities ORDER BY win_prob DESC")
        probabilities = [dict(row) for row in cursor_prob.fetchall()]
        prob_df = pd.DataFrame(probabilities)
        logging.info(f"Fetched {len(probabilities)} probability records.")

        # 2. Fetch Run Details (No Change)
        details_query = "SELECT stage, description FROM simulation_run_details WHERE simulation_id = 1 ORDER BY detail_id ASC"
        cursor_details = conn.execute(details_query)
        all_run_details = [dict(row) for row in cursor_details.fetchall()]
        logging.info(f"Fetched {len(all_run_details)} detail records to parse.")

        # --- Regex patterns (Keep as before) ---
        group_pattern = re.compile(r"^(.*?) (\d+)-(\d+) (.*?)\s+\(Elo:\s*([\d.]+)\s*vs\s*([\d.]+)\)$")
        ko_pattern = re.compile(r"^(R16|QF|SF|Final):\s*(.*?) (\d+)-(\d+) (.*?)\s+\(Winner:\s*(.*?),\s*Elo:\s*([\d.]+)\s*vs\s*([\d.]+)\)$")
        standing_pattern = re.compile(r"(\d+)\.\s*(.+?)\s*\((\d+)b,\s*([+-]?\d+)(?:,\s*(\d+)-(\d+))?\)")
        third_place_pattern = re.compile(r"(\d+)\.\s*(.+?)\s*\((\w+),(\d+)b,([+-]?\d+)\)")

        # --- First Pass: Parse Matches, KO, Standings, Qualified List ---
        logging.info("Starting FIRST parsing pass (Matches, KO, Standings)...")
        for detail in all_run_details:
            stage = detail.get('stage', 'N/A')
            description = detail.get('description', '')
            match_data = None

            if stage == 'Group Match':
                match = group_pattern.match(description)
                if match:
                    match_data = { 'team_a': match.group(1).strip(), 'score_a': int(match.group(2)), 'score_b': int(match.group(3)), 'team_b': match.group(4).strip(), 'elo_a': round(float(match.group(5))), 'elo_b': round(float(match.group(6))), 'winner': None }
                    total_group_matches = sum(len(v) for k, v in parsed_run_details['Group'].items() if isinstance(v, list))
                    round_num = (total_group_matches // 12) + 1
                    match_key = f"Match {round_num}"
                    if match_key not in parsed_run_details['Group']: parsed_run_details['Group'][match_key] = []
                    parsed_run_details['Group'][match_key].append(match_data)
            elif stage in ['R16', 'QF', 'SF', 'Final']:
                 match = ko_pattern.match(description)
                 if match:
                     match_data = { 'stage': match.group(1), 'team_a': match.group(2).strip(), 'score_a': int(match.group(3)), 'score_b': int(match.group(4)), 'team_b': match.group(5).strip(), 'winner': match.group(6).strip(), 'elo_a': round(float(match.group(7))), 'elo_b': round(float(match.group(8))) }
                     if stage in parsed_run_details: parsed_run_details[stage].append(match_data)
            elif stage == 'Qualified R16 List':
                try:
                    loaded_list = json.loads(description)
                    if isinstance(loaded_list, list) and len(loaded_list) == 16:
                        qualified_r16_teams = loaded_list
                        logging.info(f"Successfully loaded qualified R16 teams: {qualified_r16_teams}")
                    else: logging.warning(f"Loaded qualified teams list is not a list of 16: {loaded_list}")
                except Exception as e: logging.error(f"Error processing qualified teams list JSON '{description}': {e}")
            elif stage.endswith(" Standing"):
                group_name_from_stage = stage.replace(" Standing", "").strip()
                # Store group standings with a standardized key format
                # Extract the group letter - assuming the last character/word is the letter
                group_letter = group_name_from_stage.split()[-1]
                standardized_key = f"Group {group_letter}"
                
                logging.info(f"Parsing standings for '{group_name_from_stage}', standardized key: '{standardized_key}'")
                standings_list = []
                all_matches = standing_pattern.findall(description)
                successful_standings_parses = 0
                for match_tuple in all_matches:
                    try:
                        rank, team, points, gd, gf_str, ga_str = match_tuple
                        gd_val = int(gd)
                        gf_val = int(gf_str) if gf_str else None
                        ga_val = int(ga_str) if ga_str else None
                        if gf_val is not None and ga_val is None: ga_val = gf_val - gd_val
                        elif gf_val is None and ga_val is not None: gf_val = ga_val + gd_val
                        gf_val = gf_val if gf_val is not None else 0
                        ga_val = ga_val if ga_val is not None else 0
                        stats = {'rank': int(rank), 'team': team.strip(), 'P': int(points), 'GD': gd_val, 'GF': gf_val, 'GA': ga_val}
                        standings_list.append(stats)
                        successful_standings_parses += 1
                    except (ValueError, TypeError) as parse_err:
                         logging.error(f"ValueError/TypeError parsing standing entry tuple '{match_tuple}' in {group_name_from_stage}: {parse_err}")
                    except Exception as e:
                         logging.error(f"Unexpected error parsing standing entry tuple '{match_tuple}' in {group_name_from_stage}: {e}")
                if standings_list:
                    # Store with both the original key and the standardized key for robustness
                    parsed_standings[group_name_from_stage] = sorted(standings_list, key=lambda x: x['rank'])
                    parsed_standings[standardized_key] = sorted(standings_list, key=lambda x: x['rank'])
                    # Also store by just the letter for even more flexibility
                    parsed_standings[group_letter] = sorted(standings_list, key=lambda x: x['rank'])
                    logging.info(f"Parsed {successful_standings_parses} standings entries for {group_name_from_stage}")
                else:
                     logging.warning(f"Failed to parse any standings entries using findall for stage '{stage}' with description '{description}'")
        logging.info("Finished FIRST parsing pass.")
        
        # Log the keys in parsed_standings for debugging
        logging.info(f"Available keys in parsed_standings: {list(parsed_standings.keys())}")

        # --- Second Pass: Parse 3rd Place Ranking using the populated parsed_standings ---
        logging.info("Starting SECOND parsing pass (3rd Place Ranking)...")
        if not parsed_standings:
             logging.error("Cannot parse 3rd place teams because parsed_standings dictionary is empty after first pass!")
        else:
             for detail in all_run_details:
                 stage = detail.get('stage', 'N/A')
                 description = detail.get('description', '')
                 if stage == '3rd Place Ranking':
                     logging.info(f"Attempting to parse 3rd place ranking string using findall: '{description}'")
                     third_place_ranking_details = [] # Reset before parsing
                     all_third_matches = third_place_pattern.findall(description)
                     successful_parses = 0
                     for match_tuple in all_third_matches:
                         try:
                             rank, team, group_letter, points, gd = match_tuple
                             logging.debug(f"Regex matched 3rd place: rank={rank}, team={team}, group={group_letter}, P={points}, GD={gd}")
                             gf, ga = 'N/A', 'N/A'
                             
                             # Try multiple key formats for more robust lookup
                             key_alternatives = [
                                 f"Group {group_letter}",  # Standard format "Group A"
                                 f"Skupina {group_letter}", # Czech format "Skupina A"
                                 group_letter              # Just the letter "A"
                             ]
                             
                             target_team_name = team.strip()
                             found_key = None
                             for key in key_alternatives:
                                 if key in parsed_standings:
                                     found_key = key
                                     break
                                     
                             if found_key:
                                 logging.debug(f"Found standings using key: '{found_key}'")
                                 group_data = parsed_standings[found_key]
                                 team_stats_in_group = next((s for s in group_data if s.get('team') == target_team_name), None)
                                 if team_stats_in_group:
                                     logging.debug(f"Found team_stats_in_group: {team_stats_in_group}")
                                     gf_val = team_stats_in_group.get('GF')
                                     ga_val = team_stats_in_group.get('GA')
                                     gf = gf_val if gf_val is not None else 'N/A'
                                     ga = ga_val if ga_val is not None else 'N/A'
                                     logging.debug(f"Retrieved GF={gf}, GA={ga}")
                                 else: 
                                     logging.warning(f"Could not find team '{target_team_name}' in parsed standings for key '{found_key}' to get GF/GA.")
                                     # Try a case-insensitive search as a fallback
                                     team_stats_in_group = next((s for s in group_data if s.get('team').lower() == target_team_name.lower()), None)
                                     if team_stats_in_group:
                                         logging.debug(f"Found team using case-insensitive match: {team_stats_in_group}")
                                         gf_val = team_stats_in_group.get('GF')
                                         ga_val = team_stats_in_group.get('GA')
                                         gf = gf_val if gf_val is not None else 'N/A'
                                         ga = ga_val if ga_val is not None else 'N/A'
                             else: 
                                 logging.warning(f"Could not find any matching key for group '{group_letter}' when looking for GF/GA for 3rd place {target_team_name}.")
                                 # Log available keys for debugging
                                 logging.debug(f"Available keys: {list(parsed_standings.keys())}")
                                 
                             third_place_ranking_details.append({'rank': int(rank), 'team': target_team_name, 'group': group_letter, 'P': int(points), 'GD': int(gd), 'GF': gf, 'GA': ga})
                             successful_parses += 1
                         except ValueError as ve: logging.error(f"ValueError converting parsed 3rd place data for tuple '{match_tuple}': {ve}")
                         except Exception as e: logging.error(f"Unexpected error parsing 3rd place tuple '{match_tuple}': {e}")
                     logging.info(f"Successfully parsed {successful_parses} 3rd place entries using findall.")
                     third_place_ranking_details.sort(key=lambda x: x['rank'])
                     if not third_place_ranking_details: logging.warning("third_place_ranking_details list is empty after parsing attempts.")
                     break # Stop after processing the 3rd place ranking entry
        logging.info("Finished SECOND parsing pass.")

        # Final checks after both passes
        if not qualified_r16_teams: logging.warning("Qualified R16 teams list is still empty after parsing.")
        if not parsed_standings: logging.warning("parsed_standings dictionary is still empty after parsing.")
        if not third_place_ranking_details: logging.warning("third_place_ranking_details is still empty after parsing.")


        # 3. Generate Charts (No Change needed here)
        # --- Generate Win Probability Chart ---
        if not prob_df.empty:
            try:
                teams_sorted = prob_df['nationality'].tolist(); win_probs_sorted = prob_df['win_prob'].tolist()
                fig_win = go.Figure(data=[go.Bar(x=teams_sorted, y=win_probs_sorted, name='Výhra v turnaji (%)', marker_color='indianred')])
                fig_win.update_layout(title='Pravděpodobnost celkového vítězství v turnaji (%)', xaxis_title='Tým', yaxis_title='Pravděpodobnost (%)', xaxis_tickangle=-45)
                win_prob_chart_json = json.dumps(fig_win, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e: logging.error(f"Error generating win probability chart: {e}")

        # --- ELO Data Processing ---
        elo_snapshots = []
        initial_elos = {}
        final_elos = {}
        elo_df_all = pd.DataFrame()
        try:
            elo_query = "SELECT nationality, stage, elo_after_match, match_order, match_description FROM elo_snapshots WHERE simulation_id = 1 ORDER BY match_order ASC"
            cursor_elo = conn.execute(elo_query)
            elo_snapshots = [dict(row) for row in cursor_elo.fetchall()]
            if elo_snapshots:
                elo_df_all = pd.DataFrame(elo_snapshots)
                initial_df = elo_df_all[elo_df_all['match_order'] == 0]
                initial_elos = pd.Series(initial_df.elo_after_match.values, index=initial_df.nationality).to_dict()
                idx = elo_df_all.groupby('nationality')['match_order'].idxmax()
                final_df = elo_df_all.loc[idx]
                final_elos = pd.Series(final_df.elo_after_match.values, index=final_df.nationality).to_dict()
            logging.info(f"Fetched {len(elo_snapshots)} ELO snapshots. Initial: {len(initial_elos)}, Final: {len(final_elos)}")
        except Exception as e: logging.error(f"Error fetching/processing ELO snapshots: {e}")

        # --- Generate Single Elo Evolution Chart ---
        if not elo_df_all.empty and qualified_r16_teams:
            try:
                fig_elo_evolution = go.Figure()
                qualified_teams_df = elo_df_all[elo_df_all['nationality'].isin(qualified_r16_teams)].copy()
                if not qualified_teams_df.empty:
                    def map_order_to_label(order):
                        if order == 0: return "Initial";
                        elif 1 <= order <= 12: return "Group R1";
                        elif 13 <= order <= 24: return "Group R2";
                        elif 25 <= order <= 36: return "Group R3";
                        elif 37 <= order <= 44: return "R16";
                        elif 45 <= order <= 48: return "Quarter Final";
                        elif 49 <= order <= 50: return "Semi Final";
                        elif order == 51: return "Final";
                        else: return f"Unknown ({order})"
                    qualified_teams_df['agg_label'] = qualified_teams_df['match_order'].apply(map_order_to_label)
                    label_order = ["Initial", "Group R1", "Group R2", "Group R3", "R16", "Quarter Final", "Semi Final", "Final"]
                    present_labels = qualified_teams_df['agg_label'].unique(); x_axis_labels = [label for label in label_order if label in present_labels]
                    plotted_teams_count = 0
                    for team in qualified_r16_teams:
                        team_data = qualified_teams_df[qualified_teams_df['nationality'] == team].copy()
                        if not team_data.empty:
                            team_data.sort_values('match_order', inplace=True)
                            y_values_agg = []; hover_texts = []
                            for label in x_axis_labels:
                                label_data = team_data[team_data['agg_label'] == label]
                                if not label_data.empty:
                                    last_match_in_stage = label_data.iloc[-1]
                                    y_values_agg.append(last_match_in_stage['elo_after_match'])
                                    match_desc = last_match_in_stage.get('match_description', 'N/A'); stage = last_match_in_stage.get('stage', ''); hover_text = f"({match_desc})" if stage != 'Initial' else "Initial Rating"
                                    if ' vs ' in match_desc: opponents = [t.strip() for t in match_desc.split(' vs ') if t.strip() != team]; hover_text = f"vs {opponents[0]}" if opponents else hover_text
                                    hover_texts.append(hover_text)
                                else: y_values_agg.append(None); hover_texts.append(None)
                            if any(y is not None for y in y_values_agg):
                                fig_elo_evolution.add_trace(go.Scatter(x=x_axis_labels, y=y_values_agg, mode='lines+markers', name=team, text=hover_texts, hovertemplate=(f"<b>{team}</b><br>Elo: %{{y:.0f}}<br>%{{text}}<br>Stage/Round: %{{x}}<extra></extra>"), connectgaps=False))
                                plotted_teams_count += 1
                    if plotted_teams_count > 0:
                        fig_elo_evolution.update_layout(title='Vývoj ELO týmů kvalifikovaných do R16 (po fázích/kolech)', xaxis_title='Fáze / Kolo', yaxis_title='ELO Rating', xaxis_tickangle=-45, height=600, showlegend=True)
                        fig_elo_evolution.update_xaxes(categoryorder='array', categoryarray=x_axis_labels)
                        elo_evolution_chart_json = json.dumps(fig_elo_evolution, cls=plotly.utils.PlotlyJSONEncoder)
                        logging.info(f"ELO Evolution chart generated for {plotted_teams_count} teams.")
                    else: logging.warning("No teams plotted for ELO evolution chart.")
            except Exception as e: logging.error(f"Error generating ELO evolution chart: {e}"); logging.exception("Traceback:")

        # --- Generate Stacked Probability Chart ---
        if not prob_df.empty:
            try:
                prob_df['win_only'] = prob_df['win_prob']; prob_df['final_only'] = prob_df['final_prob'] - prob_df['win_prob']; prob_df['semi_only'] = prob_df['semi_prob'] - prob_df['final_prob']; prob_df['quarter_only'] = prob_df['quarter_prob'] - prob_df['semi_prob']
                prob_df[['win_only', 'final_only', 'semi_only', 'quarter_only']] = prob_df[['win_only', 'final_only', 'semi_only', 'quarter_only']].clip(lower=0)
                prob_df_sorted = prob_df.sort_values(by='quarter_prob', ascending=False)
                fig_stacked_prob = go.Figure()
                fig_stacked_prob.add_trace(go.Bar(name='Reach QF (only)', x=prob_df_sorted['nationality'], y=prob_df_sorted['quarter_only'], marker_color='lightblue'))
                fig_stacked_prob.add_trace(go.Bar(name='Reach SF (only)', x=prob_df_sorted['nationality'], y=prob_df_sorted['semi_only'], marker_color='lightgreen'))
                fig_stacked_prob.add_trace(go.Bar(name='Reach Final (only)', x=prob_df_sorted['nationality'], y=prob_df_sorted['final_only'], marker_color='gold'))
                fig_stacked_prob.add_trace(go.Bar(name='Win Tournament', x=prob_df_sorted['nationality'], y=prob_df_sorted['win_only'], marker_color='indianred'))
                fig_stacked_prob.update_layout(barmode='stack', title='Pravděpodobnost dosažení fází turnaje (%)', xaxis_title='Tým', yaxis_title='Pravděpodobnost (%)', xaxis_tickangle=-45, legend_title="Fáze")
                stacked_prob_chart_json = json.dumps(fig_stacked_prob, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e: logging.error(f"Error generating stacked probability chart: {e}")

        # --- Generate Initial vs Final ELO Chart ---
        if initial_elos and final_elos:
            try:
                logging.info("Generating Initial vs Final ELO chart...")
                teams = sorted(initial_elos.keys())
                initial_vals = [round(initial_elos.get(t, 0)) for t in teams]
                final_vals = [round(final_elos.get(t, initial_elos.get(t, 0))) for t in teams]
                fig_elo_comp = go.Figure(data=[ go.Bar(name='Initial ELO', x=teams, y=initial_vals, marker_color='blue'), go.Bar(name='Final ELO (Sim 0)', x=teams, y=final_vals, marker_color='red')])
                fig_elo_comp.update_layout(barmode='group', title='Porovnání ELO: Začátek vs. Konec (první simulace)', xaxis_title='Tým', yaxis_title='ELO Rating', xaxis_tickangle=-45, legend_title="Stav ELO")
                elo_comparison_chart_json = json.dumps(fig_elo_comp, cls=plotly.utils.PlotlyJSONEncoder)
                logging.info("Initial vs Final ELO chart generated.")
            except Exception as e: logging.error(f"Error generating Initial vs Final ELO chart: {e}")


    except sqlite3.Error as db_err:
        logging.error(f"Database error in /simulace route: {db_err}")
        abort(500)
    except Exception as e:
        logging.error(f"General error in /simulace route: {e}")
        logging.exception("Traceback:")
        abort(500)

    # Pass the required data to the template
    return render_template('simulation_results.html',
                           title='Výsledky Simulace',
                           probabilities=probabilities,
                           parsed_run_details=parsed_run_details,
                           parsed_standings=parsed_standings, # Pass the parsed standings
                           third_place_ranking=third_place_ranking_details, # Pass the parsed 3rd place list
                           group_definitions=group_definitions,
                           win_prob_chart_json=win_prob_chart_json,
                           elo_evolution_chart_json=elo_evolution_chart_json,
                           stacked_prob_chart_json=stacked_prob_chart_json,
                           elo_comparison_chart_json=elo_comparison_chart_json,
                           structured_knockout_data={'R16': parsed_run_details['R16'],
                                                     'QF': parsed_run_details['QF'],
                                                     'SF': parsed_run_details['SF'],
                                                     'Final': parsed_run_details['Final']}
                           )




# --- PLAYER COMPARISON ROUTE (UPDATED FOR FILTERING FIX) ---
@app.route('/players')
def player_comparison():
    """
    Route to display player statistics and compare two players.
    Allows filtering players by nationality and position, while preserving selections.
    Includes attribute grouping and radar chart generation.
    """
    conn = get_db()
    all_nationalities = []
    all_positions = []
    players_for_dropdown = []
    player1_data = None
    player2_data = None
    comparison_data_grouped = None
    radar_chart_json = "{}" # Initialize radar chart JSON

    selected_nationalities = request.args.getlist('nationality_filter')
    selected_positions = request.args.getlist('position_filter')
    player1_name = request.args.get('player1')
    player2_name = request.args.get('player2')

    try:
        # Fetch filter options and players for dropdown (logic remains the same)
        cursor_nats = conn.execute("SELECT DISTINCT nationality FROM players ORDER BY nationality")
        all_nationalities = [row['nationality'] for row in cursor_nats.fetchall()]
        cursor_pos = conn.execute("SELECT DISTINCT primary_position FROM players WHERE primary_position IS NOT NULL AND primary_position != '' ORDER BY primary_position")
        all_positions = [row['primary_position'] for row in cursor_pos.fetchall()]

        query = "SELECT DISTINCT player_name, nationality FROM players WHERE 1=1"
        params = []
        if selected_nationalities:
            placeholders = ','.join('?' for _ in selected_nationalities)
            query += f" AND nationality IN ({placeholders})"
            params.extend(selected_nationalities)
        if selected_positions:
            placeholders = ','.join('?' for _ in selected_positions)
            query += f" AND primary_position IN ({placeholders})"
            params.extend(selected_positions)
        query += " ORDER BY player_name"
        cursor_filtered = conn.execute(query, params)
        filtered_players_list = [dict(row) for row in cursor_filtered.fetchall()]
        filtered_player_names = {p['player_name'] for p in filtered_players_list}
        selected_players_to_add = []
        if player1_name and player1_name not in filtered_player_names:
            cursor_sel1 = conn.execute("SELECT DISTINCT player_name, nationality FROM players WHERE player_name = ?", (player1_name,))
            sel1_row = cursor_sel1.fetchone()
            if sel1_row: selected_players_to_add.append(dict(sel1_row))
        if player2_name and player2_name not in filtered_player_names and player2_name != player1_name:
            cursor_sel2 = conn.execute("SELECT DISTINCT player_name, nationality FROM players WHERE player_name = ?", (player2_name,))
            sel2_row = cursor_sel2.fetchone()
            if sel2_row: selected_players_to_add.append(dict(sel2_row))
        players_for_dropdown_dict = {p['player_name']: p for p in filtered_players_list}
        for p in selected_players_to_add: players_for_dropdown_dict[p['player_name']] = p
        players_for_dropdown = sorted(list(players_for_dropdown_dict.values()), key=lambda x: x['player_name'])

        # Fetch full data for selected players (same as before)
        if player1_name:
            cursor_p1 = conn.execute("SELECT * FROM players WHERE player_name = ?", (player1_name,))
            p1_row = cursor_p1.fetchone()
            if p1_row: player1_data = dict(p1_row)
        if player2_name:
            cursor_p2 = conn.execute("SELECT * FROM players WHERE player_name = ?", (player2_name,))
            p2_row = cursor_p2.fetchone()
            if p2_row: player2_data = dict(p2_row)

        # Perform comparison, grouping, AND radar chart generation if both players have data
        if player1_data and player2_data:
            logging.info(f"Performing comparison & chart gen for {player1_name} and {player2_name}")

            # --- Define Groups (used for both table and radar) ---
            # Using the 6 categories requested by the user
            attribute_groups = {
                "Attacking": ['offensive_awareness', 'finishing', 'kicking_power', 'heading'],
                "Ball Control": ['ball_control', 'dribbling', 'tight_possession'],
                "Passing": ['low_pass', 'lofted_pass', 'curl', 'set_piece_taking'],
                "Defending": ['defensive_awareness', 'tackling', 'aggression', 'defensive_engagement'],
                "Physical": ['speed', 'acceleration', 'jumping', 'physical_contact', 'balance', 'stamina'],
                "Goalkeeping": ['gk_awareness', 'gk_catching', 'gk_parrying', 'gk_reflexes', 'gk_reach']
            }
            radar_categories = ["Attacking", "Ball Control", "Passing", "Defending", "Physical", "Goalkeeping"]

            # --- Helper function for comparison (same as before) ---
            def compare_attribute(attr, p1_data, p2_data):
                if attr in p1_data and attr in p2_data:
                    try:
                        val1 = pd.to_numeric(p1_data[attr], errors='coerce')
                        val2 = pd.to_numeric(p2_data[attr], errors='coerce')
                        if pd.notna(val1) and pd.notna(val2):
                            p1_better = val1 > val2
                            p2_better = val2 > val1
                            diff = abs(round(val1 - val2, 1))
                            return { 'name': attr.replace('_', ' ').title(), 'p1_value': int(val1) if val1 == int(val1) else val1, 'p2_value': int(val2) if val2 == int(val2) else val2, 'p1_better': p1_better, 'p2_better': p2_better, 'diff': int(diff) if diff == int(diff) else diff }
                        else:
                             return {'name': attr.replace('_', ' ').title(), 'p1_value': p1_data[attr], 'p2_value': p2_data[attr], 'p1_better': False, 'p2_better': False, 'diff': 0}
                    except Exception as e:
                        logging.error(f"Error comparing attribute '{attr}': {e}")
                        return {'name': attr.replace('_', ' ').title(), 'p1_value': 'Error', 'p2_value': 'Error', 'p1_better': False, 'p2_better': False, 'diff': 0}
                else: return None

            # --- Generate Grouped Data for Table ---
            comparison_data_grouped = []
            processed_attributes_table = set()
            overall_comp = compare_attribute('overall_rating', player1_data, player2_data)
            if overall_comp:
                 comparison_data_grouped.append({"group_name": "Overall", "attributes": [overall_comp]})
                 processed_attributes_table.add('overall_rating')

            for group_name, attrs_in_group in attribute_groups.items():
                group_results = []
                for attr in attrs_in_group:
                    comp_result = compare_attribute(attr, player1_data, player2_data)
                    if comp_result:
                        group_results.append(comp_result)
                        processed_attributes_table.add(attr)
                if group_results:
                    comparison_data_grouped.append({"group_name": group_name, "attributes": group_results})

            # --- Calculate Averages for Radar Chart ---
            radar_values_p1 = []
            radar_values_p2 = []

            for category in radar_categories:
                attrs_in_category = attribute_groups.get(category, [])
                p1_vals = []
                p2_vals = []
                for attr in attrs_in_category:
                    if attr in player1_data:
                        val1 = pd.to_numeric(player1_data[attr], errors='coerce')
                        if pd.notna(val1): p1_vals.append(val1)
                    if attr in player2_data:
                        val2 = pd.to_numeric(player2_data[attr], errors='coerce')
                        if pd.notna(val2): p2_vals.append(val2)

                # Calculate average, handle empty list (assign 0 or other default)
                avg_p1 = np.mean(p1_vals) if p1_vals else 0
                avg_p2 = np.mean(p2_vals) if p2_vals else 0
                radar_values_p1.append(round(avg_p1)) # Round average for display
                radar_values_p2.append(round(avg_p2))

            # --- Generate Radar Chart JSON ---
            try:
                fig_radar = go.Figure()

                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_values_p1,
                    theta=radar_categories,
                    fill='toself',
                    name=player1_name,
                    line_color='blue'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_values_p2,
                    theta=radar_categories,
                    fill='toself',
                    name=player2_name,
                    line_color='red'
                ))

                fig_radar.update_layout(
                  polar=dict(
                    radialaxis=dict(
                      visible=True,
                      range=[0, 100] # Assuming attributes are roughly 0-100 scale
                    )),
                  showlegend=True,
                  title="Radar Chart Comparison",
                  margin=dict(l=40, r=40, t=80, b=40) # Adjust margins
                )
                radar_chart_json = json.dumps(fig_radar, cls=plotly.utils.PlotlyJSONEncoder)
                logging.info("Radar chart JSON generated.")
            except Exception as e:
                logging.error(f"Error generating radar chart: {e}")
                radar_chart_json = "{}" # Ensure it's valid JSON even on error

    except sqlite3.Error as e:
        logging.error(f"Database error in /players route: {e}")
        abort(500)
    except Exception as e:
        logging.error(f"General error in /players route: {e}")
        abort(500)

    # Render the template
    return render_template('player_comparison.html',
                           title='Porovnání hráčů',
                           all_nationalities=all_nationalities,
                           all_positions=all_positions,
                           selected_nationalities=selected_nationalities,
                           selected_positions=selected_positions,
                           all_players=players_for_dropdown,
                           player1_name=player1_name,
                           player2_name=player2_name,
                           player1_data=player1_data,
                           player2_data=player2_data,
                           comparison_data_grouped=comparison_data_grouped, # Pass grouped data
                           radar_chart_json=radar_chart_json) # *** Pass radar chart JSON ***

@app.route('/analysis/descriptive')
def descriptive_analysis():
    conn = get_db()
    team_stats = []
    all_teams_list = []
    team_comparison_data = None
    single_team_analysis_results = None
    available_attributes = _get_available_attributes() # Use helper
    selected_attributes_for_single = []
    selected_include_gk = 'yes'
    page_title = "Deskriptivní Analýza Týmů" # Base title
    view_mode = 'overview' # Default view mode

    # Get parameters from URL
    current_scope_param = request.args.get('scope') # For overview table scope
    selected_team1 = request.args.get('team1')
    selected_team2 = request.args.get('team2')
    comparison_scope = request.args.get('comparison_scope', 'all') # Scope for comparison/single
    selected_attributes_for_single = request.args.getlist('attributes')
    selected_include_gk = request.args.get('include_gk', 'yes')

    # Determine view mode based on selected teams
    if selected_team1 and not selected_team2:
        view_mode = 'single_team'
    elif selected_team1 and selected_team2:
        view_mode = 'two_team_comparison'
    # else: view_mode remains 'overview'

    # Fetch team list for dropdowns (always needed)
    try:
        cursor_teams = conn.execute("SELECT DISTINCT nationality FROM rosters ORDER BY nationality")
        all_teams_list = [row['nationality'] for row in cursor_teams.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching team list: {e}")
        all_teams_list = [] # Ensure it's an empty list on error

    # Process data based on view mode
    try:
        if view_mode == 'single_team':
            page_title = f"Detailní Analýza Týmu: {selected_team1}"
            single_team_analysis_results = _get_single_team_analysis(conn, selected_team1, comparison_scope, selected_attributes_for_single, selected_include_gk)

        elif view_mode == 'two_team_comparison':
            page_title = f"Porovnání Týmů: {selected_team1} vs {selected_team2}"
            team_comparison_data = _get_two_team_comparison_data(conn, selected_team1, selected_team2, comparison_scope)

        elif view_mode == 'overview':
            # Use the scope from URL param if present, otherwise default to 'all'
            current_scope = current_scope_param if current_scope_param in ['squad', 'all'] else 'all'
            team_stats, page_title_suffix = _get_overview_stats(conn, current_scope)
            page_title += page_title_suffix

        # Set current_scope for template buttons (needed even if view changes)
        # This ensures overview buttons show correct active state after comparison/single view
        current_scope = current_scope_param if current_scope_param in ['squad', 'all'] else 'all'


    except sqlite3.Error as e:
        logging.error(f"Database error: {e}"); team_stats, team_comparison_data, single_team_analysis_results = [], None, None; page_title = "Chyba databáze"; view_mode = 'error'
    except Exception as e:
        logging.exception(f"General error in descriptive_analysis:"); team_stats, team_comparison_data, single_team_analysis_results = [], None, None; page_title = "Chyba serveru"; view_mode = 'error'

    # Render Template
    return render_template('descriptive_analysis.html',
                           title=page_title,
                           view_mode=view_mode,
                           team_stats=team_stats,
                           current_scope=current_scope,
                           all_teams=all_teams_list,
                           selected_team1=selected_team1,
                           selected_team2=selected_team2,
                           comparison_scope=comparison_scope,
                           team_comparison_data=team_comparison_data,
                           single_team_analysis_results=single_team_analysis_results,
                           available_attributes=available_attributes,
                           selected_attributes_for_single=selected_attributes_for_single,
                           selected_include_gk=selected_include_gk
                           )


@app.route('/analysis/descriptive/export')
def descriptive_analysis_export():
    conn = get_db()
    # Get all relevant parameters from the request
    scope = request.args.get('scope', 'all') # For overview export
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    comparison_scope = request.args.get('comparison_scope', 'all') # For comparison/single export
    attributes_str = request.args.get('attributes', '') # Comma-separated list from GET param
    attributes = attributes_str.split(',') if attributes_str else []
    include_gk = request.args.get('include_gk', 'yes')

    # Determine which data to export based on parameters
    view_mode = 'overview'
    if team1 and not team2: view_mode = 'single_team'
    elif team1 and team2: view_mode = 'two_team_comparison'

    df_export = pd.DataFrame() # Initialize empty dataframe
    filename = "descriptive_analysis_export.csv" # Default filename

    try:
        if view_mode == 'single_team':
            filename = f"analysis_{team1}_{comparison_scope}{'_noGK' if include_gk == 'no' else ''}.csv"
            results_dict = _get_single_team_analysis(conn, team1, comparison_scope, attributes, include_gk)
            # --- Transform single_team_analysis_results dict into a list of dicts for CSV ---
            export_data = []
            if results_dict:
                 # Ensure attributes requested are used if available, otherwise use dict keys
                attrs_to_export = attributes if attributes else list(results_dict.keys())
                valid_attrs_to_export = [attr for attr in attrs_to_export if attr in results_dict]

                for attr_key in valid_attrs_to_export:
                    data = results_dict[attr_key]
                    if 'error' not in data:
                         # Row for Max
                         export_data.append({
                             'Attribute': data.get('name', attr_key.replace('_', ' ').title()),
                             'Statistic': 'Max Value',
                             'Value': data.get('max_val', 'N/A'),
                             'Player(s)': ', '.join(data.get('max_players', [])) if data.get('max_players') else ''
                         })
                         # Row for Min
                         export_data.append({
                             'Attribute': data.get('name', attr_key.replace('_', ' ').title()),
                             'Statistic': 'Min Value',
                             'Value': data.get('min_val', 'N/A'),
                             'Player(s)': ', '.join(data.get('min_players', [])) if data.get('min_players') else ''
                         })
                    else: # Handle error case for attribute
                         export_data.append({
                             'Attribute': data.get('name', attr_key.replace('_', ' ').title()),
                             'Statistic': 'Error',
                             'Value': data.get('error'),
                             'Player(s)': ''
                         })
            if export_data: df_export = pd.DataFrame(export_data)

        elif view_mode == 'two_team_comparison':
            filename = f"comparison_{team1}_vs_{team2}_{comparison_scope}.csv"
            results_list = _get_two_team_comparison_data(conn, team1, team2, comparison_scope)
            if results_list:
                 # Rename columns for clarity in CSV
                 df = pd.DataFrame(results_list)
                 df_export = df.rename(columns={
                     'name': 'Attribute',
                     't1_value': f'{team1} Avg',
                     't2_value': f'{team2} Avg',
                     't1_better': f'{team1} > {team2}',
                     't2_better': f'{team2} > {team1}',
                     'diff': 'Abs Difference'
                 })


        elif view_mode == 'overview':
            filename = f"overview_stats_{scope}.csv"
            results_list, _ = _get_overview_stats(conn, scope) # Ignore title suffix
            if results_list:
                 # Rename columns for clarity
                 df = pd.DataFrame(results_list)
                 df_export = df.rename(columns={
                     'nationality': 'Nationality',
                     'player_count': f'Player Count ({scope.capitalize()})',
                     'avg_age': 'Avg Age',
                     'avg_height': 'Avg Height (cm)',
                     'avg_weight': 'Avg Weight (kg)',
                     'avg_rating': 'Avg Rating'
                 })

        # Prepare CSV response
        if not df_export.empty:
            csv_buffer = StringIO()
            df_export.to_csv(csv_buffer, index=False, encoding='utf-8') # Use utf-8 encoding
            csv_buffer.seek(0)
            response = make_response(csv_buffer.getvalue())
            response.mimetype = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        else:
            logging.warning(f"No data generated for export view '{view_mode}' with current parameters.")
            # Return a simple message or redirect back with a flash message (more advanced)
            return "Could not generate export data for the selected criteria.", 404

    except Exception as e:
        logging.exception("Error during CSV export generation:")
        return "An error occurred while generating the CSV export.", 500

@app.route('/players/export')
def player_comparison_export():
    """
    Handles the export of player data (single or comparison) to CSV.
    Removes 'Difference' and 'Better' columns from comparison export.
    Includes fix for non-ASCII characters in filenames and SyntaxError fix.
    """
    conn = get_db()
    player1_name = request.args.get('player1')
    player2_name = request.args.get('player2')

    df_export = pd.DataFrame()
    filename_raw = "player_export.csv" # Raw filename for internal use/logging
    filename_ascii = "player_export.csv" # Fallback ASCII filename

    try:
        cursor = conn.cursor()
        player1_data = None
        player2_data = None

        # Fetch data for player 1
        if player1_name:
            cursor.execute("SELECT * FROM players WHERE player_name = ?", (player1_name,))
            p1_row = cursor.fetchone()
            if p1_row:
                player1_data = dict(p1_row)
            else:
                logging.warning(f"Export: Player 1 '{player1_name}' not found.")
                return "Player 1 not found.", 404

        # Fetch data for player 2 if provided
        if player2_name:
            cursor.execute("SELECT * FROM players WHERE player_name = ?", (player2_name,))
            p2_row = cursor.fetchone()
            if p2_row:
                player2_data = dict(p2_row)
            else:
                logging.warning(f"Export: Player 2 '{player2_name}' not found. Exporting Player 1 only.")
                player2_name = None # Reset player2_name

        # --- Generate DataFrame based on whether it's single or comparison ---
        if player1_data and player2_data:
            # --- Two Player Comparison Export (Modified) ---
            p1_name_safe = ''.join(c for c in player1_name if ord(c) < 128).replace(' ','_').replace('.','').replace('+','')[:20]
            p2_name_safe = ''.join(c for c in player2_name if ord(c) < 128).replace(' ','_').replace('.','').replace('+','')[:20]
            filename_raw = f"comparison_{player1_name}_vs_{player2_name}.csv"
            filename_ascii = f"comparison_{p1_name_safe}_vs_{p2_name_safe}.csv"
            logging.info(f"Exporting comparison (no diff/better) for {player1_name} and {player2_name}")

            comparison_export_list = []
            # Use all available attributes from player 1 data dict keys (excluding name/nationality for comparison itself)
            attributes_to_compare = [k for k in player1_data.keys() if k not in ['player_name', 'nationality', 'team_name']] # Adjust excluded keys as needed

            for attr in attributes_to_compare:
                val1_raw = player1_data.get(attr)
                val2_raw = player2_data.get(attr)
                # Keep original values for display
                comp_entry = {
                    'Attribute': attr.replace('_', ' ').title(),
                    f'{player1_name}': val1_raw,
                    f'{player2_name}': val2_raw,
                    # Removed 'Difference' and 'Better' keys/calculations
                }
                comparison_export_list.append(comp_entry)

            if comparison_export_list:
                df_export = pd.DataFrame(comparison_export_list)

        elif player1_data:
            # --- Single Player Export (Unchanged) ---
            p1_name_safe = ''.join(c for c in player1_name if ord(c) < 128).replace(' ','_').replace('.','').replace('+','')[:30]
            filename_raw = f"player_{player1_name}.csv"
            filename_ascii = f"player_{p1_name_safe}.csv"
            logging.info(f"Exporting data for single player: {player1_name}")
            single_export_list = [{'Attribute': key.replace('_', ' ').title(), 'Value': value} for key, value in player1_data.items()]
            if single_export_list:
                df_export = pd.DataFrame(single_export_list)

        # --- Generate CSV Response (Unchanged header logic) ---
        if not df_export.empty:
            csv_buffer = StringIO()
            df_export.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_buffer.seek(0)
            response = make_response(csv_buffer.getvalue())
            response.mimetype = 'text/csv; charset=utf-8'

            try:
                filename_encoded = urllib.parse.quote(filename_raw, safe='')
                disposition = f"attachment; filename=\"{filename_ascii}\"; filename*=UTF-8''{filename_encoded}"
                response.headers['Content-Disposition'] = disposition
                logging.info(f"CSV export '{filename_raw}' generated successfully with header: {disposition}")
            except Exception as header_err:
                 logging.error(f"Error encoding filename for header: {header_err}. Using ASCII fallback.")
                 response.headers['Content-Disposition'] = f'attachment; filename="{filename_ascii}"'

            return response
        else:
            logging.warning(f"No data generated for player export with parameters: player1={player1_name}, player2={player2_name}")
            return "Could not generate export data for the selected player(s).", 404

    except sqlite3.Error as e:
        logging.error(f"Database error during player export: {e}")
        return "Database error occurred during export.", 500
    except Exception as e:
        logging.exception("General error during player export:")
        return "An error occurred while generating the CSV export.", 500


# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    # --- (No changes here) ---
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    # --- (No changes here) ---
    logging.exception("Internal Server Error:")
    return render_template('500.html'), 500
