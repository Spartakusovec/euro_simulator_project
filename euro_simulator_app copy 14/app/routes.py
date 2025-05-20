from flask import (
    render_template,
    current_app as app,
    g,
    abort,
    url_for,
    request,
    make_response,
    jsonify,
)
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE = os.path.join(BASE_DIR, "data", "database.db")
ATTRIBUTE_GROUPS = {
    "Physical": [
        "height",
        "weight",
        "age",
        "speed",
        "acceleration",
        "jumping",
        "physical_contact",
        "balance",
        "stamina",
    ],
    "Attacking": ["offensive_awareness", "finishing", "kicking_power", "heading"],
    "Ball Control": ["ball_control", "dribbling", "tight_possession"],
    "Passing": ["low_pass", "lofted_pass", "curl", "set_piece_taking"],
    "Defending": [
        "defensive_awareness",
        "tackling",
        "aggression",
        "defensive_engagement",
    ],
    "Goalkeeping": [
        "gk_awareness",
        "gk_catching",
        "gk_parrying",
        "gk_reflexes",
        "gk_reach",
    ],
    "Other": ["overall_rating"],
}
position_categories = {
    "Attacker": [
        "CF",
        "SS",
        "LWF",
        "RWF",
        "Centre Forward",
        "Second Striker",
        "Left Wing Forward",
        "Right Wing Forward",
    ],
    "Midfielder": [
        "AMF",
        "CMF",
        "DMF",
        "LMF",
        "RMF",
        "Attacking Midfielder",
        "Centre Midfielder",
        "Defensive Midfielder",
        "Left Midfielder",
        "Right Midfielder",
    ],
    "Defender": ["CB", "LB", "RB", "Centre Back", "Left Back", "Right Back"],
    "Goalkeeper": ["GK", "Goalkeeper"],
}
formations = {
    "4-4-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "4-3-3": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Centre Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Left Wing Forward",
        "Right Wing Forward",
        "Centre Forward",
    ],
    "3-5-2": [
        "Goalkeeper",
        "Centre Back",
        "Centre Back",
        "Centre Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "4-2-3-1": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Defensive Midfielder",
        "Defensive Midfielder",
        "Attacking Midfielder",
        "Left Midfielder",
        "Right Midfielder",
        "Centre Forward",
    ],
    "4-1-4-1": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Defensive Midfielder",
        "Left Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Centre Forward",
    ],
    "5-3-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Centre Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Second Striker",
        "Centre Forward",
    ],
    "3-4-3": [
        "Goalkeeper",
        "Centre Back",
        "Centre Back",
        "Centre Back",
        "Right Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Left Midfielder",
        "Right Wing Forward",
        "Centre Forward",
        "Left Wing Forward",
    ],
    "4-4-1-1": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
    ],
    "4-3-1-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "4-2-2-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Defensive Midfielder",
        "Defensive Midfielder",
        "Attacking Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "4-3-2-1": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Attacking Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
    ],
    "3-5-1-1": [
        "Goalkeeper",
        "Centre Back",
        "Centre Back",
        "Centre Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Second Striker",
        "Centre Forward",
    ],
    "4-1-3-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Defensive Midfielder",
        "Left Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Second Striker",
        "Centre Forward",
    ],
    "4-4-2 Diamond": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Left Midfielder",
        "Right Midfielder",
        "Defensive Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "5-2-1-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Defensive Midfielder",
        "Defensive Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "4-3-3 (Attack)": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Centre Midfielder",
        "Centre Midfielder",
        "Attacking Midfielder",
        "Left Wing Forward",
        "Right Wing Forward",
        "Centre Forward",
    ],
    "4-1-2-1-2": [
        "Goalkeeper",
        "Left Back",
        "Centre Back",
        "Centre Back",
        "Right Back",
        "Defensive Midfielder",
        "Left Midfielder",
        "Right Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
        "Centre Forward",
    ],
    "3-4-2-1": [
        "Goalkeeper",
        "Centre Back",
        "Centre Back",
        "Centre Back",
        "Left Midfielder",
        "Centre Midfielder",
        "Centre Midfielder",
        "Right Midfielder",
        "Attacking Midfielder",
        "Attacking Midfielder",
        "Centre Forward",
    ],
}
groups = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Slovenia", "Denmark", "Serbia", "England"],
    "D": ["Poland", "Netherlands", "Austria", "France"],
    "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "F": ["Turkey", "Georgia", "Portugal", "Czechia"],
}
COUNTRY_CODES = {
    "Germany": "DE",
    "Scotland": "GB-SCT",
    "Hungary": "HU",
    "Switzerland": "CH",
    "Spain": "ES",
    "Croatia": "HR",
    "Italy": "IT",
    "Albania": "AL",
    "Slovenia": "SI",
    "Denmark": "DK",
    "Serbia": "RS",
    "England": "GB-ENG",
    "Poland": "PL",
    "Netherlands": "NL",
    "Austria": "AT",
    "France": "FR",
    "Belgium": "BE",
    "Slovakia": "SK",
    "Romania": "RO",
    "Ukraine": "UA",
    "Türkiye": "TR",
    "Georgia": "GE",
    "Portugal": "PT",
    "Czechia": "CZ",
    "Turkey": "TR",
}


def get_flag_emoji(country_name):
    code = COUNTRY_CODES.get(country_name)
    if not code or len(code) != 2:
        if country_name == "England":
            return "🏴󠁧󠁢󠁥󠁮󠁧󠁿"
        if country_name == "Scotland":
            return "🏴󠁧󠁢󠁳󠁣󠁴󠁿"
        return "🏳️"
    return chr(ord("🇦") + ord(code[0]) - ord("A")) + chr(
        ord("🇦") + ord(code[1]) - ord("A")
    )


def get_database_path():
    if "BASE_DIR" not in app.config:
        app.config["BASE_DIR"] = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    return os.path.join(app.config["BASE_DIR"], "data", "database.db")


def get_db():
    if "db" not in g:
        db_path = get_database_path()
        try:
            g.db = sqlite3.connect(db_path)
            g.db.row_factory = sqlite3.Row
            logging.info(f"DB connection successful to {db_path}")
        except sqlite3.Error as e:
            logging.error(f"DB connection error: {e}")
            abort(500)
    return g.db


def _get_overview_stats(conn, scope):
    team_stats = []
    page_title_suffix = ""
    numeric_cols_desc = ["age", "height", "weight", "overall_rating"]
    df_desc = None
    query = ""

    if scope == "squad":
        query = """ SELECT p.nationality, p.age, p.height, p.weight, p.overall_rating FROM players p JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality """
        page_title_suffix = " (Soupiska)"
    else:
        scope = "all"
        query = f"SELECT nationality, {', '.join(numeric_cols_desc)} FROM players"
        page_title_suffix = " (Všichni hráči)"

    try:
        df_desc = pd.read_sql_query(query, conn)
        if df_desc is not None and not df_desc.empty:
            for col in numeric_cols_desc:
                df_desc[col] = pd.to_numeric(df_desc[col], errors="coerce")
            df_desc.dropna(subset=numeric_cols_desc, how="all", inplace=True)
            if not df_desc.empty:
                grouped_stats = (
                    df_desc.groupby("nationality")[numeric_cols_desc]
                    .agg(
                        avg_age=("age", "mean"),
                        avg_height=("height", "mean"),
                        avg_weight=("weight", "mean"),
                        avg_rating=("overall_rating", "mean"),
                        player_count=("overall_rating", "size"),
                    )
                    .reset_index()
                )
                for col in ["avg_age", "avg_height", "avg_weight", "avg_rating"]:
                    grouped_stats[col] = grouped_stats[col].round(1)
                team_stats = grouped_stats.to_dict("records")
                logging.info(
                    f"Overview helper calculated stats for {len(team_stats)} teams."
                )
            else:
                logging.warning(
                    f"Overview helper: No valid numeric data for scope '{scope}'."
                )
        else:
            logging.warning(f"Overview helper: No data fetched for scope '{scope}'.")
    except Exception as e:
        logging.error(f"Error in _get_overview_stats: {e}")
        team_stats = []

    return team_stats, page_title_suffix


def _get_two_team_comparison_data(conn, team1, team2, scope):
    team_comparison_data = None
    attributes_to_compare = [
        "overall_rating",
        "offensive_awareness",
        "ball_control",
        "dribbling",
        "tight_possession",
        "low_pass",
        "lofted_pass",
        "finishing",
        "heading",
        "set_piece_taking",
        "curl",
        "defensive_awareness",
        "tackling",
        "aggression",
        "defensive_engagement",
        "speed",
        "acceleration",
        "kicking_power",
        "jumping",
        "physical_contact",
        "balance",
        "stamina",
        "gk_awareness",
        "gk_catching",
        "gk_parrying",
        "gk_reflexes",
        "gk_reach",
    ]
    calculated_averages = {}
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(players)")
        actual_player_cols = {info[1] for info in cursor.fetchall()}
        cols_to_select_comp = [
            attr for attr in attributes_to_compare if attr in actual_player_cols
        ]
        cols_str_comp_quoted = ", ".join([f'p."{col}"' for col in cols_to_select_comp])

        for team in [team1, team2]:
            df_team = None
            query_base_comp = f"SELECT {cols_str_comp_quoted} FROM players p "
            if scope == "squad":
                query_team = (
                    query_base_comp
                    + "JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality WHERE p.nationality = ?"
                )
            else:
                query_team = query_base_comp + "WHERE p.nationality = ?"
            df_team = pd.read_sql_query(query_team, conn, params=(team,))
            if df_team is not None and not df_team.empty:
                team_averages = {
                    attr: (
                        round(pd.to_numeric(df_team[attr], errors="coerce").mean(), 1)
                        if pd.notna(
                            pd.to_numeric(df_team[attr], errors="coerce").mean()
                        )
                        else None
                    )
                    for attr in cols_to_select_comp
                    if attr in df_team.columns
                }
                calculated_averages[team] = team_averages
            else:
                calculated_averages[team] = {attr: None for attr in cols_to_select_comp}

        team1_avg_stats = calculated_averages.get(team1)
        team2_avg_stats = calculated_averages.get(team2)
        if team1_avg_stats and team2_avg_stats:
            team_comparison_data = []
            for attr in attributes_to_compare:
                val1 = team1_avg_stats.get(attr)
                val2 = team2_avg_stats.get(attr)
                comp_entry = {
                    "name": attr.replace("_", " ").title(),
                    "t1_value": val1 if pd.notna(val1) else "N/A",
                    "t2_value": val2 if pd.notna(val2) else "N/A",
                    "t1_better": False,
                    "t2_better": False,
                    "diff": 0,
                }
                if pd.notna(val1) and pd.notna(val2):
                    comp_entry["t1_better"] = val1 > val2
                    comp_entry["t2_better"] = val2 > val1
                    diff = abs(round(val1 - val2, 1))
                    comp_entry["diff"] = int(diff) if diff == int(diff) else diff
                team_comparison_data.append(comp_entry)
    except Exception as e:
        logging.error(f"Error in _get_two_team_comparison_data: {e}")
        team_comparison_data = None

    return team_comparison_data


def _get_single_team_analysis(
    conn, team_name, scope, attributes_to_analyze, include_gk
):
    analysis_results = {}
    available_attributes = _get_available_attributes()
    numeric_attrs_in_db = []

    try:
        base_cols = ["player_name", "nationality", "primary_position"]
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(players)")
        actual_cols = {info[1] for info in cursor.fetchall()}
        numeric_attrs_in_db = [
            attr
            for attr in available_attributes
            if attr not in base_cols and attr in actual_cols
        ]
        valid_attrs_to_analyze = [
            attr for attr in attributes_to_analyze if attr in numeric_attrs_in_db
        ]
        if not valid_attrs_to_analyze:
            valid_attrs_to_analyze = [
                attr
                for attr in [
                    "height",
                    "weight",
                    "age",
                    "overall_rating",
                    "speed",
                    "acceleration",
                    "finishing",
                    "tackling",
                    "dribbling",
                    "low_pass",
                    "stamina",
                ]
                if attr in numeric_attrs_in_db
            ]

        all_cols_to_select = list(dict.fromkeys(base_cols + valid_attrs_to_analyze))
        cols_str_select_quoted = ", ".join([f'p."{col}"' for col in all_cols_to_select])

        df_single_team = None
        query_base = f"SELECT {cols_str_select_quoted} FROM players p "
        if scope == "squad":
            query_single = (
                query_base
                + "JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality WHERE p.nationality = ?"
            )
        else:
            query_single = query_base + "WHERE p.nationality = ?"
        df_single_team = pd.read_sql_query(query_single, conn, params=(team_name,))

        if df_single_team is not None and not df_single_team.empty:
            df_analysis_subset = df_single_team.copy()
            if include_gk == "no" and "primary_position" in df_analysis_subset.columns:
                condition = (
                    df_analysis_subset["primary_position"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    != "gk"
                )
                df_analysis_subset = df_analysis_subset.loc[condition].copy()

            if not df_analysis_subset.empty:
                for attr in valid_attrs_to_analyze:
                    if attr in df_analysis_subset.columns:
                        df_analysis_subset[attr] = pd.to_numeric(
                            df_analysis_subset[attr], errors="coerce"
                        )
                        valid_data = df_analysis_subset.dropna(subset=[attr])
                        if not valid_data.empty:
                            max_val, min_val = (
                                valid_data[attr].max(),
                                valid_data[attr].min(),
                            )
                            max_players, min_players = ["N/A"], ["N/A"]
                            if "player_name" in valid_data.columns:
                                max_players = valid_data.loc[
                                    valid_data[attr] == max_val, "player_name"
                                ].tolist()
                                min_players = valid_data.loc[
                                    valid_data[attr] == min_val, "player_name"
                                ].tolist()
                            analysis_results[attr] = {
                                "name": attr.replace("_", " ").title(),
                                "max_val": (
                                    int(max_val)
                                    if pd.notna(max_val) and max_val == int(max_val)
                                    else (
                                        round(max_val, 1)
                                        if pd.notna(max_val)
                                        else "N/A"
                                    )
                                ),
                                "max_players": max_players,
                                "min_val": (
                                    int(min_val)
                                    if pd.notna(min_val) and min_val == int(min_val)
                                    else (
                                        round(min_val, 1)
                                        if pd.notna(min_val)
                                        else "N/A"
                                    )
                                ),
                                "min_players": min_players,
                            }
                        else:
                            analysis_results[attr] = {
                                "name": attr.replace("_", " ").title(),
                                "error": "No valid data",
                            }
                    else:
                        analysis_results[attr] = {
                            "name": attr.replace("_", " ").title(),
                            "error": "Attr. unavailable",
                        }
            else:
                for attr in valid_attrs_to_analyze:
                    analysis_results[attr] = {
                        "name": attr.replace("_", " ").title(),
                        "error": "No players match criteria",
                    }
        else:
            logging.warning(f"Single team helper: No data fetched for {team_name}")
            analysis_results = {}
    except Exception as e:
        logging.error(f"Error in _get_single_team_analysis: {e}")
        analysis_results = {}

    return analysis_results


def _get_available_attributes():
    available_attributes = [
        "age",
        "height",
        "weight",
        "overall_rating",
        "offensive_awareness",
        "ball_control",
        "dribbling",
        "tight_possession",
        "low_pass",
        "lofted_pass",
        "finishing",
        "heading",
        "set_piece_taking",
        "curl",
        "defensive_awareness",
        "tackling",
        "aggression",
        "defensive_engagement",
        "speed",
        "acceleration",
        "kicking_power",
        "jumping",
        "physical_contact",
        "balance",
        "stamina",
        "gk_awareness",
        "gk_catching",
        "gk_parrying",
        "gk_reflexes",
        "gk_reach",
    ]
    available_attributes.sort()
    return available_attributes


def _build_position_maps(conn):
    position_map_full_to_abbr = {}
    position_map_abbr_to_full = {}
    try:
        cursor_map = conn.execute(
            """
            SELECT DISTINCT p.primary_position, r.assigned_position
            FROM players p
            JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality
            WHERE p.primary_position IS NOT NULL AND p.primary_position != ''
              AND r.assigned_position IS NOT NULL AND r.assigned_position != ''
        """
        )
        for row in cursor_map.fetchall():
            full_name = row["assigned_position"]
            abbr_name = row["primary_position"]
            if full_name and abbr_name:
                if full_name not in position_map_full_to_abbr:
                    position_map_full_to_abbr[full_name] = abbr_name
                if abbr_name not in position_map_abbr_to_full:
                    position_map_abbr_to_full[abbr_name] = full_name

        explicit_abbr_to_full = {
            "GK": "Goalkeeper",
            "CB": "Centre Back",
            "LB": "Left Back",
            "RB": "Right Back",
            "DMF": "Defensive Midfielder",
            "CMF": "Centre Midfielder",
            "LMF": "Left Midfielder",
            "RMF": "Right Midfielder",
            "AMF": "Attacking Midfielder",
            "LWF": "Left Wing Forward",
            "RWF": "Right Wing Forward",
            "SS": "Second Striker",
            "CF": "Centre Forward",
        }
        position_map_abbr_to_full.update(explicit_abbr_to_full)
        for abbr, full in position_map_abbr_to_full.items():
            position_map_full_to_abbr[full] = abbr

        logging.info(
            f"Built position maps: Full->Abbr ({len(position_map_full_to_abbr)}), Abbr->Full ({len(position_map_abbr_to_full)})"
        )

    except sqlite3.Error as e:
        logging.error(f"Database error building position maps: {e}", exc_info=True)
    return position_map_full_to_abbr, position_map_abbr_to_full


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()
        logging.info("DB connection closed.")


@app.route("/")
@app.route("/index")
def index():
    conn = get_db()
    teams_data = []
    try:
        cursor = conn.execute(
            "SELECT DISTINCT nationality FROM rosters ORDER BY nationality"
        )
        teams_raw = [row["nationality"] for row in cursor.fetchall()]
        teams_data = [
            {"name": team, "flag": get_flag_emoji(team)} for team in teams_raw
        ]
    except Exception as e:
        logging.error(f"Error fetching teams: {e}")
    return render_template(
        "index.html", title="Domovská stránka", teams_data=teams_data
    )


@app.route("/tym/<team_name>")
def team_roster(team_name):
    conn = get_db()
    players_list = []
    formation_name = "N/A"
    total_team_rating = 0
    try:
        cursor = conn.execute(
            "SELECT player_name, assigned_position, overall_rating_in_position, formation_name FROM rosters WHERE nationality = ?",
            (team_name,),
        )
        roster_data_raw = [dict(row) for row in cursor.fetchall()]
        if not roster_data_raw:
            abort(404)
        formation_name = roster_data_raw[0]["formation_name"]
        total_team_rating = sum(
            p["overall_rating_in_position"] for p in roster_data_raw
        )
        if formation_name in formations:
            pos_map = {pos: i for i, pos in enumerate(formations[formation_name])}
            players_list = sorted(
                roster_data_raw, key=lambda p: pos_map.get(p["assigned_position"], 99)
            )
        else:
            players_list = sorted(roster_data_raw, key=lambda p: p["player_name"])
    except Exception as e:
        logging.error(f"Error reading roster for {team_name}: {e}")
        abort(500)
    return render_template(
        "team_roster.html",
        title=f"Soupiska - {team_name}",
        team_name=team_name,
        formation_name=formation_name,
        players=players_list,
        total_team_rating=total_team_rating,
    )


@app.route("/api/filtered_players")
def get_filtered_players():
    conn = get_db()
    players_for_dropdown = []
    position_map_full_to_abbr, _ = _build_position_maps(conn)

    selected_nationalities = request.args.getlist("nationality_filter")
    selected_positions_full = request.args.getlist("position_filter")

    logging.info(
        f"API Request: Nats={selected_nationalities}, Pos (Full)={selected_positions_full}"
    )

    selected_positions_abbr = []
    if selected_positions_full:
        for full_name in selected_positions_full:
            abbr = position_map_full_to_abbr.get(full_name)
            if abbr:
                selected_positions_abbr.append(abbr)
            else:
                logging.warning(
                    f"API: Could not map position '{full_name}' to an abbreviation."
                )
        logging.info(
            f"API: Translated positions to abbreviations: {selected_positions_abbr}"
        )

    try:
        query = """
            SELECT DISTINCT p.player_name, p.nationality
            FROM players p
            WHERE 1=1
        """
        params = []

        if selected_nationalities:
            placeholders = ",".join("?" for _ in selected_nationalities)
            query += f" AND p.nationality IN ({placeholders})"
            params.extend(selected_nationalities)

        if selected_positions_abbr:
            placeholders = ",".join("?" for _ in selected_positions_abbr)
            query += f" AND p.primary_position IN ({placeholders})"
            params.extend(selected_positions_abbr)
        elif selected_positions_full:
            logging.warning(
                "API: Position filter requested but no valid abbreviations found. Ignoring position filter."
            )

        query += " ORDER BY p.player_name"

        cursor_filtered = conn.execute(query, params)
        players_for_dropdown = [
            {
                "player_name": row["player_name"],
                "nationality": row["nationality"],
                "flag": get_flag_emoji(row["nationality"]),
            }
            for row in cursor_filtered.fetchall()
        ]
        logging.info(f"API Response: Found {len(players_for_dropdown)} players.")

    except sqlite3.Error as e:
        logging.error(f"Database error in /api/filtered_players: {e}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        logging.error(f"General error in /api/filtered_players: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

    return jsonify(players_for_dropdown)


@app.route("/simulace")
def simulation_results():
    conn = get_db()
    probabilities = []
    parsed_run_details = {
        "Group": {"Match 1": [], "Match 2": [], "Match 3": []},
        "R16": [],
        "QF": [],
        "SF": [],
        "Final": [],
    }
    third_place_ranking_details = []
    win_prob_chart_json = "{}"
    elo_evolution_chart_json = "{}"
    qualified_r16_teams = []
    parsed_standings = {}
    stacked_prob_chart_json = "{}"
    elo_comparison_chart_json = "{}"
    group_definitions = groups

    try:
        cursor_prob = conn.execute(
            "SELECT * FROM simulation_probabilities ORDER BY win_prob DESC"
        )
        probabilities = [dict(row) for row in cursor_prob.fetchall()]
        prob_df = pd.DataFrame(probabilities)
        logging.info(f"Fetched {len(probabilities)} probability records.")

        details_query = "SELECT stage, description FROM simulation_run_details WHERE simulation_id = 1 ORDER BY detail_id ASC"
        cursor_details = conn.execute(details_query)
        all_run_details = [dict(row) for row in cursor_details.fetchall()]
        logging.info(f"Fetched {len(all_run_details)} detail records to parse.")

        group_pattern = re.compile(
            r"^(.*?) (\d+)-(\d+) (.*?)\s+\(Elo:\s*([\d.]+)\s*vs\s*([\d.]+)\)$"
        )
        ko_pattern = re.compile(
            r"^(R16|QF|SF|Final)\s*(?:\((M\d+|QF\d+|SF\d+)\))?:\s*"
            r"(.*?)\s+(\d+)-(\d+)\s+(.*?)\s+"
            r"\(Winner:\s*(.*?),\s*Elo:\s*([\d.]+)\s*vs\s*([\d.]+)\)$"
        )
        standing_pattern = re.compile(
            r"(\d+)\.\s*(.+?)\s*\((\d+)b,\s*([+-]?\d+)(?:,\s*(\d+)-(\d+))?\)"
        )
        third_place_pattern = re.compile(
            r"(\d+)\.\s*(.+?)\s*\((\w+),(\d+)b,([+-]?\d+)\)"
        )

        logging.info("Starting FIRST parsing pass (Matches, KO, Standings)...")
        for detail in all_run_details:
            stage = detail.get("stage", "N/A")
            description = detail.get("description", "")
            match_data = None

            if stage == "Group Match":
                match = group_pattern.match(description)
                if match:
                    match_data = {
                        "team_a": match.group(1).strip(),
                        "score_a": int(match.group(2)),
                        "score_b": int(match.group(3)),
                        "team_b": match.group(4).strip(),
                        "elo_a": round(float(match.group(5))),
                        "elo_b": round(float(match.group(6))),
                        "winner": None,
                    }

                    total_group_matches = sum(
                        len(v)
                        for k, v in parsed_run_details["Group"].items()
                        if isinstance(v, list)
                    )
                    round_num = (total_group_matches // 12) + 1
                    match_key = f"Match {round_num}"
                    if match_key not in parsed_run_details["Group"]:
                        parsed_run_details["Group"][match_key] = []
                    parsed_run_details["Group"][match_key].append(match_data)
            elif stage in ["R16", "QF", "SF", "Final"]:

                match = ko_pattern.match(description)
                if match:

                    match_stage = match.group(1)
                    match_id = match.group(2)
                    team_a = match.group(3).strip()
                    score_a = int(match.group(4))
                    score_b = int(match.group(5))
                    team_b = match.group(6).strip()
                    winner = match.group(7).strip()
                    elo_a = round(float(match.group(8)))
                    elo_b = round(float(match.group(9)))

                    match_data = {
                        "stage": match_stage,
                        "match_id": match_id,
                        "team_a": team_a,
                        "score_a": score_a,
                        "score_b": score_b,
                        "team_b": team_b,
                        "winner": winner,
                        "elo_a": elo_a,
                        "elo_b": elo_b,
                    }

                    if match_stage in parsed_run_details:
                        parsed_run_details[match_stage].append(match_data)
                    else:
                        logging.warning(
                            f"Parsed KO match stage '{match_stage}' not found in parsed_run_details keys."
                        )
                else:
                    logging.warning(
                        f"Could not parse KO description with new pattern: '{description}'"
                    )

            elif stage == "Qualified R16 List":
                try:
                    loaded_list = json.loads(description)
                    if isinstance(loaded_list, list) and len(loaded_list) == 16:
                        qualified_r16_teams = loaded_list
                        logging.info(
                            f"Successfully loaded qualified R16 teams: {qualified_r16_teams}"
                        )
                    else:
                        logging.warning(
                            f"Loaded qualified teams list is not a list of 16: {loaded_list}"
                        )
                except Exception as e:
                    logging.error(
                        f"Error processing qualified teams list JSON '{description}': {e}"
                    )
            elif stage.endswith(" Standing"):
                group_name_from_stage = stage.replace(" Standing", "").strip()
                group_letter = group_name_from_stage.split()[-1]
                standardized_key = f"Group {group_letter}"

                logging.info(
                    f"Parsing standings for '{group_name_from_stage}', standardized key: '{standardized_key}'"
                )
                standings_list = []
                all_matches = standing_pattern.findall(description)
                successful_standings_parses = 0
                for match_tuple in all_matches:
                    try:
                        rank, team, points, gd, gf_str, ga_str = match_tuple
                        gd_val = int(gd)
                        gf_val = int(gf_str) if gf_str else None
                        ga_val = int(ga_str) if ga_str else None
                        if gf_val is not None and ga_val is None:
                            ga_val = gf_val - gd_val
                        elif gf_val is None and ga_val is not None:
                            gf_val = ga_val + gd_val
                        gf_val = gf_val if gf_val is not None else 0
                        ga_val = ga_val if ga_val is not None else 0
                        stats = {
                            "rank": int(rank),
                            "team": team.strip(),
                            "P": int(points),
                            "GD": gd_val,
                            "GF": gf_val,
                            "GA": ga_val,
                        }
                        standings_list.append(stats)
                        successful_standings_parses += 1
                    except (ValueError, TypeError) as parse_err:
                        logging.error(
                            f"ValueError/TypeError parsing standing entry tuple '{match_tuple}' in {group_name_from_stage}: {parse_err}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Unexpected error parsing standing entry tuple '{match_tuple}' in {group_name_from_stage}: {e}"
                        )
                if standings_list:
                    parsed_standings[group_name_from_stage] = sorted(
                        standings_list, key=lambda x: x["rank"]
                    )
                    parsed_standings[standardized_key] = sorted(
                        standings_list, key=lambda x: x["rank"]
                    )
                    parsed_standings[group_letter] = sorted(
                        standings_list, key=lambda x: x["rank"]
                    )
                    logging.info(
                        f"Parsed {successful_standings_parses} standings entries for {group_name_from_stage}"
                    )
                else:
                    logging.warning(
                        f"Failed to parse any standings entries using findall for stage '{stage}' with description '{description}'"
                    )
        logging.info("Finished FIRST parsing pass.")

        logging.info(
            f"Available keys in parsed_standings: {list(parsed_standings.keys())}"
        )

        logging.info("Starting SECOND parsing pass (3rd Place Ranking)...")
        if not parsed_standings:
            logging.error(
                "Cannot parse 3rd place teams because parsed_standings dictionary is empty after first pass!"
            )
        else:
            for detail in all_run_details:
                stage = detail.get("stage", "N/A")
                description = detail.get("description", "")
                if stage == "3rd Place Ranking":
                    logging.info(
                        f"Attempting to parse 3rd place ranking string using findall: '{description}'"
                    )
                    third_place_ranking_details = []
                    all_third_matches = third_place_pattern.findall(description)
                    successful_parses = 0
                    for match_tuple in all_third_matches:
                        try:
                            rank, team, group_letter, points, gd = match_tuple
                            logging.debug(
                                f"Regex matched 3rd place: rank={rank}, team={team}, group={group_letter}, P={points}, GD={gd}"
                            )
                            gf, ga = "N/A", "N/A"

                            key_alternatives = [
                                f"Group {group_letter}",
                                f"Skupina {group_letter}",
                                group_letter,
                            ]
                            target_team_name = team.strip()
                            found_key = None
                            for key in key_alternatives:
                                if key in parsed_standings:
                                    found_key = key
                                    break

                            if found_key:
                                logging.debug(
                                    f"Found standings using key: '{found_key}'"
                                )
                                group_data = parsed_standings[found_key]
                                team_stats_in_group = next(
                                    (
                                        s
                                        for s in group_data
                                        if s.get("team") == target_team_name
                                    ),
                                    None,
                                )
                                if team_stats_in_group:
                                    logging.debug(
                                        f"Found team_stats_in_group: {team_stats_in_group}"
                                    )
                                    gf_val = team_stats_in_group.get("GF")
                                    ga_val = team_stats_in_group.get("GA")
                                    gf = gf_val if gf_val is not None else "N/A"
                                    ga = ga_val if ga_val is not None else "N/A"
                                else:
                                    logging.warning(
                                        f"Could not find stats for team '{target_team_name}' within group data for key '{found_key}'."
                                    )
                            else:
                                logging.warning(
                                    f"Could not find any matching key for group '{group_letter}' when looking for GF/GA for 3rd place {target_team_name}."
                                )
                                logging.debug(
                                    f"Available keys: {list(parsed_standings.keys())}"
                                )

                            third_place_ranking_details.append(
                                {
                                    "rank": int(rank),
                                    "team": target_team_name,
                                    "group": group_letter,
                                    "P": int(points),
                                    "GD": int(gd),
                                    "GF": gf,
                                    "GA": ga,
                                }
                            )  # Include GA
                            successful_parses += 1
                        except ValueError as ve:
                            logging.error(
                                f"ValueError converting parsed 3rd place data for tuple '{match_tuple}': {ve}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Unexpected error parsing 3rd place tuple '{match_tuple}': {e}"
                            )
                    logging.info(
                        f"Successfully parsed {successful_parses} 3rd place entries using findall."
                    )
                    third_place_ranking_details.sort(key=lambda x: x["rank"])
                    if not third_place_ranking_details:
                        logging.warning(
                            "third_place_ranking_details list is empty after parsing attempts."
                        )
                    break
        logging.info("Finished SECOND parsing pass.")

        if not qualified_r16_teams:
            logging.warning("Qualified R16 teams list is still empty after parsing.")
        if not parsed_standings:
            logging.warning("parsed_standings dictionary is still empty after parsing.")
        if not third_place_ranking_details:
            logging.warning("third_place_ranking_details is still empty after parsing.")

        if not prob_df.empty:
            try:
                teams_sorted = prob_df["nationality"].tolist()
                win_probs_sorted = prob_df["win_prob"].tolist()
                fig_win = go.Figure(
                    data=[
                        go.Bar(
                            x=teams_sorted,
                            y=win_probs_sorted,
                            name="Výhra v turnaji (%)",
                            marker_color="indianred",
                        )
                    ]
                )
                fig_win.update_layout(
                    title="Pravděpodobnost celkového vítězství v turnaji (%)",
                    xaxis_title="Tým",
                    yaxis_title="Pravděpodobnost (%)",
                    xaxis_tickangle=-45,
                )
                win_prob_chart_json = json.dumps(
                    fig_win, cls=plotly.utils.PlotlyJSONEncoder
                )
            except Exception as e:
                logging.error(f"Error generating win probability chart: {e}")

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
                initial_df = elo_df_all[elo_df_all["match_order"] == 0]
                initial_elos = pd.Series(
                    initial_df.elo_after_match.values, index=initial_df.nationality
                ).to_dict()
                idx = elo_df_all.groupby("nationality")["match_order"].idxmax()
                final_df = elo_df_all.loc[idx]
                final_elos = pd.Series(
                    final_df.elo_after_match.values, index=final_df.nationality
                ).to_dict()
            logging.info(
                f"Fetched {len(elo_snapshots)} ELO snapshots. Initial: {len(initial_elos)}, Final: {len(final_elos)}"
            )
        except Exception as e:
            logging.error(f"Error fetching/processing ELO snapshots: {e}")

        if not elo_df_all.empty and qualified_r16_teams:
            try:
                fig_elo_evolution = go.Figure()
                qualified_teams_df = elo_df_all[
                    elo_df_all["nationality"].isin(qualified_r16_teams)
                ].copy()
                if not qualified_teams_df.empty:

                    def map_order_to_label(order):
                        if order == 0:
                            return "Initial"
                        elif 1 <= order <= 12:
                            return "Group R1"
                        elif 13 <= order <= 24:
                            return "Group R2"
                        elif 25 <= order <= 36:
                            return "Group R3"
                        elif 37 <= order <= 44:
                            return "R16"
                        elif 45 <= order <= 48:
                            return "Quarter Final"
                        elif 49 <= order <= 50:
                            return "Semi Final"
                        elif order == 51:
                            return "Final"
                        else:
                            return f"Unknown ({order})"

                    qualified_teams_df["agg_label"] = qualified_teams_df[
                        "match_order"
                    ].apply(map_order_to_label)
                    label_order = [
                        "Initial",
                        "Group R1",
                        "Group R2",
                        "Group R3",
                        "R16",
                        "Quarter Final",
                        "Semi Final",
                        "Final",
                    ]
                    present_labels = qualified_teams_df["agg_label"].unique()
                    x_axis_labels = [
                        label for label in label_order if label in present_labels
                    ]
                    plotted_teams_count = 0
                    for team in qualified_r16_teams:
                        team_data = qualified_teams_df[
                            qualified_teams_df["nationality"] == team
                        ].copy()
                        if not team_data.empty:
                            team_data.sort_values("match_order", inplace=True)
                            y_values_agg = []
                            hover_texts = []
                            for label in x_axis_labels:
                                label_data = team_data[team_data["agg_label"] == label]
                                if not label_data.empty:
                                    last_match_in_label = label_data.iloc[-1]
                                    y_values_agg.append(
                                        round(last_match_in_label["elo_after_match"])
                                    )
                                    hover_texts.append(
                                        last_match_in_label["match_description"]
                                    )
                                else:
                                    y_values_agg.append(None)
                                    hover_texts.append("")
                            if any(y is not None for y in y_values_agg):
                                fig_elo_evolution.add_trace(
                                    go.Scatter(
                                        x=x_axis_labels,
                                        y=y_values_agg,
                                        mode="lines+markers",
                                        name=team,
                                        text=hover_texts,
                                        hovertemplate=(
                                            f"<b>{team}</b><br>Elo: %{{y:.0f}}<br>%{{text}}<br>Stage/Round: %{{x}}<extra></extra>"
                                        ),
                                        connectgaps=False,
                                    )
                                )
                                plotted_teams_count += 1
                    if plotted_teams_count > 0:
                        fig_elo_evolution.update_layout(
                            title="Vývoj ELO týmů kvalifikovaných do R16 (po fázích/kolech)",
                            xaxis_title="Fáze / Kolo",
                            yaxis_title="ELO Rating",
                            xaxis_tickangle=-45,
                            height=600,
                            showlegend=True,
                        )
                        fig_elo_evolution.update_xaxes(
                            categoryorder="array", categoryarray=x_axis_labels
                        )
                        elo_evolution_chart_json = json.dumps(
                            fig_elo_evolution, cls=plotly.utils.PlotlyJSONEncoder
                        )
                        logging.info(
                            f"ELO Evolution chart generated for {plotted_teams_count} teams."
                        )
                    else:
                        logging.warning("No teams plotted for ELO evolution chart.")
            except Exception as e:
                logging.error(f"Error generating ELO evolution chart: {e}")
                logging.exception("Traceback:")

        if not prob_df.empty:
            try:
                prob_df["win_only"] = prob_df["win_prob"]
                prob_df["final_only"] = prob_df["final_prob"] - prob_df["win_prob"]
                prob_df["semi_only"] = prob_df["semi_prob"] - prob_df["final_prob"]
                prob_df["quarter_only"] = prob_df["quarter_prob"] - prob_df["semi_prob"]
                prob_df[["win_only", "final_only", "semi_only", "quarter_only"]] = (
                    prob_df[
                        ["win_only", "final_only", "semi_only", "quarter_only"]
                    ].clip(lower=0)
                )
                prob_df_sorted = prob_df.sort_values(by="quarter_prob", ascending=False)
                fig_stacked_prob = go.Figure()
                fig_stacked_prob.add_trace(
                    go.Bar(
                        name="Reach QF (only)",
                        x=prob_df_sorted["nationality"],
                        y=prob_df_sorted["quarter_only"],
                        marker_color="lightblue",
                    )
                )
                fig_stacked_prob.add_trace(
                    go.Bar(
                        name="Reach SF (only)",
                        x=prob_df_sorted["nationality"],
                        y=prob_df_sorted["semi_only"],
                        marker_color="lightgreen",
                    )
                )
                fig_stacked_prob.add_trace(
                    go.Bar(
                        name="Reach Final (only)",
                        x=prob_df_sorted["nationality"],
                        y=prob_df_sorted["final_only"],
                        marker_color="gold",
                    )
                )
                fig_stacked_prob.add_trace(
                    go.Bar(
                        name="Win Tournament",
                        x=prob_df_sorted["nationality"],
                        y=prob_df_sorted["win_only"],
                        marker_color="indianred",
                    )
                )
                fig_stacked_prob.update_layout(
                    barmode="stack",
                    title="Pravděpodobnost dosažení fází turnaje (%)",
                    xaxis_title="Tým",
                    yaxis_title="Pravděpodobnost (%)",
                    xaxis_tickangle=-45,
                    legend_title="Fáze",
                )
                stacked_prob_chart_json = json.dumps(
                    fig_stacked_prob, cls=plotly.utils.PlotlyJSONEncoder
                )
            except Exception as e:
                logging.error(f"Error generating stacked probability chart: {e}")

        if initial_elos and final_elos:
            try:
                logging.info("Generating Initial vs Final ELO chart...")
                teams = sorted(initial_elos.keys())
                initial_vals = [round(initial_elos.get(t, 0)) for t in teams]
                final_vals = [
                    round(final_elos.get(t, initial_elos.get(t, 0))) for t in teams
                ]
                fig_elo_comp = go.Figure(
                    data=[
                        go.Bar(
                            name="Initial ELO",
                            x=teams,
                            y=initial_vals,
                            marker_color="blue",
                        ),
                        go.Bar(
                            name="Final ELO (Sim 0)",
                            x=teams,
                            y=final_vals,
                            marker_color="red",
                        ),
                    ]
                )
                fig_elo_comp.update_layout(
                    barmode="group",
                    title="Porovnání ELO: Začátek vs. Konec (první simulace)",
                    xaxis_title="Tým",
                    yaxis_title="ELO Rating",
                    xaxis_tickangle=-45,
                    legend_title="Stav ELO",
                )
                elo_comparison_chart_json = json.dumps(
                    fig_elo_comp, cls=plotly.utils.PlotlyJSONEncoder
                )
                logging.info("Initial vs Final ELO chart generated.")
            except Exception as e:
                logging.error(f"Error generating Initial vs Final ELO chart: {e}")

    except sqlite3.Error as db_err:
        logging.error(f"Database error in /simulace route: {db_err}")
        abort(500)
    except Exception as e:
        logging.error(f"General error in /simulace route: {e}")
        logging.exception("Traceback:")
        abort(500)

    structured_knockout_data = {
        "R16": parsed_run_details.get("R16", []),
        "QF": parsed_run_details.get("QF", []),
        "SF": parsed_run_details.get("SF", []),
        "Final": parsed_run_details.get("Final", []),
    }

    return render_template(
        "simulation_results.html",
        title="Výsledky Simulace",
        probabilities=probabilities,
        parsed_run_details=parsed_run_details,
        parsed_standings=parsed_standings,
        third_place_ranking=third_place_ranking_details,
        group_definitions=group_definitions,
        win_prob_chart_json=win_prob_chart_json,
        elo_evolution_chart_json=elo_evolution_chart_json,
        stacked_prob_chart_json=stacked_prob_chart_json,
        elo_comparison_chart_json=elo_comparison_chart_json,
        structured_knockout_data=structured_knockout_data,
        active_page="simulation",
    )


@app.route("/players")
def player_comparison():

    conn = get_db()
    all_nationalities = []
    all_positions_full = []
    players_for_dropdown = []
    player1_data = None
    player2_data = None
    comparison_data_grouped = None
    radar_chart_json = "{}"

    position_map_full_to_abbr, position_map_abbr_to_full = _build_position_maps(conn)

    selected_nationalities = request.args.getlist("nationality_filter")
    selected_positions_full = request.args.getlist("position_filter")
    player1_name = request.args.get("player1")
    player2_name = request.args.get("player2")

    position_categories_full = {
        "Attacker": [
            "Centre Forward",
            "Second Striker",
            "Left Wing Forward",
            "Right Wing Forward",
        ],
        "Midfielder": [
            "Attacking Midfielder",
            "Centre Midfielder",
            "Defensive Midfielder",
            "Left Midfielder",
            "Right Midfielder",
        ],
        "Defender": ["Centre Back", "Left Back", "Right Back"],
        "Goalkeeper": ["Goalkeeper"],
    }

    try:

        cursor_nats = conn.execute(
            "SELECT DISTINCT nationality FROM players ORDER BY nationality"
        )
        all_nationalities = [row["nationality"] for row in cursor_nats.fetchall()]

        cursor_pos_full = conn.execute(
            "SELECT DISTINCT assigned_position FROM rosters WHERE assigned_position IS NOT NULL AND assigned_position != '' ORDER BY assigned_position"
        )
        all_positions_full = [
            row["assigned_position"] for row in cursor_pos_full.fetchall()
        ]
        if not all_positions_full:
            logging.warning(
                "No assigned_position found in rosters. Position filter UI might be incomplete."
            )

            cursor_pos_fallback = conn.execute(
                "SELECT DISTINCT primary_position FROM players WHERE primary_position IS NOT NULL AND primary_position != '' ORDER BY primary_position"
            )
            fallback_abbrs = [
                row["primary_position"] for row in cursor_pos_fallback.fetchall()
            ]
            all_positions_full = sorted(
                list(
                    set(
                        position_map_abbr_to_full.get(abbr, abbr)
                        for abbr in fallback_abbrs
                    )
                )
            )

        selected_positions_abbr = []
        if selected_positions_full:
            for full_name in selected_positions_full:
                abbr = position_map_full_to_abbr.get(full_name)
                if abbr:
                    selected_positions_abbr.append(abbr)
                else:
                    logging.warning(
                        f"Route: Could not map position '{full_name}' to an abbreviation."
                    )
            logging.info(
                f"Route: Translated positions to abbreviations: {selected_positions_abbr}"
            )

        query = """
            SELECT DISTINCT p.player_name, p.nationality
            FROM players p
            WHERE 1=1
        """
        params = []
        if selected_nationalities:
            placeholders = ",".join("?" for _ in selected_nationalities)
            query += f" AND p.nationality IN ({placeholders})"
            params.extend(selected_nationalities)

        if selected_positions_abbr:
            placeholders = ",".join("?" for _ in selected_positions_abbr)
            query += f" AND p.primary_position IN ({placeholders})"
            params.extend(selected_positions_abbr)
        elif selected_positions_full:
            logging.warning(
                "Route: Position filter requested but no valid abbreviations found. Ignoring position filter."
            )

        query += " ORDER BY p.player_name"

        cursor_filtered = conn.execute(query, params)
        filtered_players_list = [dict(row) for row in cursor_filtered.fetchall()]
        filtered_player_names = {p["player_name"] for p in filtered_players_list}

        selected_players_to_add = []
        if player1_name and player1_name not in filtered_player_names:
            cursor_sel1 = conn.execute(
                "SELECT DISTINCT player_name, nationality FROM players WHERE player_name = ?",
                (player1_name,),
            )
            sel1_row = cursor_sel1.fetchone()
            if sel1_row:
                selected_players_to_add.append(dict(sel1_row))
        if (
            player2_name
            and player2_name not in filtered_player_names
            and player2_name != player1_name
        ):
            cursor_sel2 = conn.execute(
                "SELECT DISTINCT player_name, nationality FROM players WHERE player_name = ?",
                (player2_name,),
            )
            sel2_row = cursor_sel2.fetchone()
            if sel2_row:
                selected_players_to_add.append(dict(sel2_row))

        players_for_dropdown_dict = {p["player_name"]: p for p in filtered_players_list}
        for p in selected_players_to_add:
            players_for_dropdown_dict[p["player_name"]] = p
        players_for_dropdown = sorted(
            list(players_for_dropdown_dict.values()), key=lambda x: x["player_name"]
        )

        if player1_name:
            cursor_p1 = conn.execute(
                """
                SELECT p.*, r.assigned_position
                FROM players p
                LEFT JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality
                WHERE p.player_name = ?
            """,
                (player1_name,),
            )
            p1_row = cursor_p1.fetchone()
            if p1_row:
                player1_data = dict(p1_row)
                assigned_pos = player1_data.get("assigned_position")
                primary_abbr = player1_data.get("primary_position")
                player1_data["display_position"] = (
                    assigned_pos
                    or position_map_abbr_to_full.get(
                        primary_abbr, primary_abbr or "N/A"
                    )
                )

        if player2_name:
            cursor_p2 = conn.execute(
                """
                SELECT p.*, r.assigned_position
                FROM players p
                LEFT JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality
                WHERE p.player_name = ?
            """,
                (player2_name,),
            )
            p2_row = cursor_p2.fetchone()
            if p2_row:
                player2_data = dict(p2_row)

                assigned_pos = player2_data.get("assigned_position")
                primary_abbr = player2_data.get("primary_position")
                player2_data["display_position"] = (
                    assigned_pos
                    or position_map_abbr_to_full.get(
                        primary_abbr, primary_abbr or "N/A"
                    )
                )

        if player1_data and player2_data:
            logging.info(
                f"Performing comparison & chart gen for {player1_name} and {player2_name}"
            )

            attribute_groups = {
                "Attacking": [
                    "offensive_awareness",
                    "finishing",
                    "kicking_power",
                    "heading",
                ],
                "Ball Control": ["ball_control", "dribbling", "tight_possession"],
                "Passing": ["low_pass", "lofted_pass", "curl", "set_piece_taking"],
                "Defending": [
                    "defensive_awareness",
                    "tackling",
                    "aggression",
                    "defensive_engagement",
                ],
                "Physical": [
                    "speed",
                    "acceleration",
                    "jumping",
                    "physical_contact",
                    "balance",
                    "stamina",
                ],
                "Goalkeeping": [
                    "gk_awareness",
                    "gk_catching",
                    "gk_parrying",
                    "gk_reflexes",
                    "gk_reach",
                ],
            }
            radar_categories = [
                "Attacking",
                "Ball Control",
                "Passing",
                "Defending",
                "Physical",
                "Goalkeeping",
            ]

            def compare_attribute(attr, p1_data, p2_data):
                val1_raw = p1_data.get(attr)
                val2_raw = p2_data.get(attr)

                try:
                    val1 = pd.to_numeric(val1_raw, errors="coerce")
                except:
                    val1 = np.nan
                try:
                    val2 = pd.to_numeric(val2_raw, errors="coerce")
                except:
                    val2 = np.nan

                comp_entry = {
                    "name": attr.replace("_", " ").title(),
                    "p1_value": val1_raw if pd.notna(val1_raw) else "N/A",
                    "p2_value": val2_raw if pd.notna(val2_raw) else "N/A",
                    "p1_better": False,
                    "p2_better": False,
                    "diff": 0,
                }

                if pd.notna(val1) and pd.notna(val2):
                    comp_entry["p1_better"] = val1 > val2
                    comp_entry["p2_better"] = val2 > val1
                    diff = abs(round(val1 - val2, 1))
                    comp_entry["diff"] = int(diff) if diff == int(diff) else diff
                return comp_entry

            comparison_data_grouped = []
            processed_attributes_table = set()
            overall_comp = compare_attribute(
                "overall_rating", player1_data, player2_data
            )
            if overall_comp:
                comparison_data_grouped.append(
                    {"group_name": "Overall", "attributes": [overall_comp]}
                )
                processed_attributes_table.add("overall_rating")

            for group_name, attrs_in_group in attribute_groups.items():
                group_results = []
                for attr in attrs_in_group:
                    if (
                        attr not in processed_attributes_table
                        and attr in player1_data
                        and attr in player2_data
                    ):
                        comp = compare_attribute(attr, player1_data, player2_data)
                        if comp:
                            group_results.append(comp)
                            processed_attributes_table.add(attr)
                if group_results:
                    comparison_data_grouped.append(
                        {"group_name": group_name, "attributes": group_results}
                    )

            radar_values_p1 = []
            radar_values_p2 = []

            for category in radar_categories:
                attrs_in_category = attribute_groups.get(category, [])
                p1_vals = []
                p2_vals = []
                for attr in attrs_in_category:
                    try:
                        p1_val = pd.to_numeric(player1_data.get(attr), errors="coerce")
                    except:
                        p1_val = np.nan
                    try:
                        p2_val = pd.to_numeric(player2_data.get(attr), errors="coerce")
                    except:
                        p2_val = np.nan
                    if pd.notna(p1_val):
                        p1_vals.append(p1_val)
                    if pd.notna(p2_val):
                        p2_vals.append(p2_val)

                avg_p1 = np.mean(p1_vals) if p1_vals else 0
                avg_p2 = np.mean(p2_vals) if p2_vals else 0
                radar_values_p1.append(round(avg_p1))
                radar_values_p2.append(round(avg_p2))

            try:
                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_values_p1,
                        theta=radar_categories,
                        fill="toself",
                        name=player1_name,
                        line_color="blue",
                    )
                )
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_values_p2,
                        theta=radar_categories,
                        fill="toself",
                        name=player2_name,
                        line_color="red",
                    )
                )
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title="Radar Chart Comparison",
                    margin=dict(l=40, r=40, t=80, b=40),
                )
                radar_chart_json = json.dumps(
                    fig_radar, cls=plotly.utils.PlotlyJSONEncoder
                )
                logging.info("Radar chart JSON generated.")
            except Exception as e:
                logging.error(f"Error generating radar chart: {e}")
                radar_chart_json = "{}"

    except sqlite3.Error as e:
        logging.error(f"Database error in /players route: {e}")
        abort(500)
    except Exception as e:
        logging.error(f"General error in /players route: {e}")
        abort(500)

    return render_template(
        "player_comparison.html",
        title="Porovnání hráčů",
        all_nationalities=all_nationalities,
        all_positions=all_positions_full,
        position_categories=position_categories_full,
        selected_nationalities=selected_nationalities,
        selected_positions=selected_positions_full,
        all_players=players_for_dropdown,
        player1_name=player1_name,
        player2_name=player2_name,
        player1_data=player1_data,
        player2_data=player2_data,
        comparison_data_grouped=comparison_data_grouped,
        radar_chart_json=radar_chart_json,
        position_map=position_map_abbr_to_full,
        active_page="comparison",
        get_flag_emoji=get_flag_emoji,
    )


@app.route("/analysis/descriptive")
def descriptive_analysis():
    conn = get_db()
    team_stats = []
    all_teams_list = []
    team_comparison_data = None
    single_team_analysis_results = None
    available_attributes = _get_available_attributes()
    selected_attributes_for_single = []
    selected_include_gk = "yes"
    page_title = "Deskriptivní Analýza Týmů"
    view_mode = "overview"

    current_scope_param = request.args.get("scope")
    selected_team1 = request.args.get("team1")
    selected_team2 = request.args.get("team2")
    comparison_scope = request.args.get("comparison_scope", "all")
    selected_attributes_for_single = request.args.getlist("attributes")
    selected_include_gk = request.args.get("include_gk", "yes")

    if selected_team1 and not selected_team2:
        view_mode = "single_team"
    elif selected_team1 and selected_team2:
        view_mode = "two_team_comparison"

    try:
        cursor_teams = conn.execute(
            "SELECT DISTINCT nationality FROM rosters ORDER BY nationality"
        )
        all_teams_list = [row["nationality"] for row in cursor_teams.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching team list: {e}")
        all_teams_list = []

    try:
        if view_mode == "single_team":
            page_title = f"Detailní Analýza Týmu: {selected_team1}"

            valid_selected_attrs = [
                attr
                for attr in selected_attributes_for_single
                if attr in available_attributes
            ]
            single_team_analysis_results = _get_single_team_analysis(
                conn,
                selected_team1,
                comparison_scope,
                valid_selected_attrs,
                selected_include_gk,
            )

        elif view_mode == "two_team_comparison":
            page_title = f"Porovnání Týmů: {selected_team1} vs {selected_team2}"
            team_comparison_data = _get_two_team_comparison_data(
                conn, selected_team1, selected_team2, comparison_scope
            )

        elif view_mode == "overview":

            current_scope = (
                current_scope_param
                if current_scope_param in ["squad", "all"]
                else "all"
            )
            team_stats, page_title_suffix = _get_overview_stats(conn, current_scope)
            page_title += page_title_suffix

        current_scope = (
            current_scope_param if current_scope_param in ["squad", "all"] else "all"
        )

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        team_stats, team_comparison_data, single_team_analysis_results = [], None, None
        page_title = "Chyba databáze"
        view_mode = "error"
    except Exception as e:
        logging.exception(f"General error in descriptive_analysis:")
        team_stats, team_comparison_data, single_team_analysis_results = [], None, None
        page_title = "Chyba serveru"
        view_mode = "error"

    attributes_by_group_for_template = {}
    if view_mode == "single_team":
        for group, attrs in ATTRIBUTE_GROUPS.items():
            group_attrs_available = [
                attr for attr in attrs if attr in available_attributes
            ]
            if group_attrs_available:
                attributes_by_group_for_template[group] = group_attrs_available
        grouped_attrs_set = {
            attr for group_list in ATTRIBUTE_GROUPS.values() for attr in group_list
        }
        ungrouped_available = [
            attr for attr in available_attributes if attr not in grouped_attrs_set
        ]
        if ungrouped_available:
            attributes_by_group_for_template["Other"] = ungrouped_available

    return render_template(
        "descriptive_analysis.html",
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
        attributes_by_group=attributes_by_group_for_template,
        selected_attributes_for_single=selected_attributes_for_single,
        selected_include_gk=selected_include_gk,
        get_flag_emoji=get_flag_emoji,
        active_page="analysis",
    )


@app.route("/analysis/descriptive/export")
def descriptive_analysis_export():
    conn = get_db()
    scope = request.args.get("scope", "all")
    team1 = request.args.get("team1")
    team2 = request.args.get("team2")
    comparison_scope = request.args.get("comparison_scope", "all")
    attributes_str = request.args.get("attributes", "")
    attributes = attributes_str.split(",") if attributes_str else []
    include_gk = request.args.get("include_gk", "yes")

    view_mode = "overview"
    if team1 and not team2:
        view_mode = "single_team"
    elif team1 and team2:
        view_mode = "two_team_comparison"

    df_export = pd.DataFrame()
    filename = "descriptive_analysis_export.csv"

    try:
        if view_mode == "single_team":
            filename = f"analysis_{team1}_{comparison_scope}{'_noGK' if include_gk == 'no' else ''}.csv"
            results_dict = _get_single_team_analysis(
                conn, team1, comparison_scope, attributes, include_gk
            )
            export_data = []
            if results_dict:
                attrs_to_export = (
                    attributes if attributes else list(results_dict.keys())
                )

                valid_attrs_to_export = [
                    attr for attr in attrs_to_export if attr in results_dict
                ]

                for attr_key in valid_attrs_to_export:
                    data = results_dict[attr_key]
                    if "error" not in data:

                        export_data.append(
                            {
                                "Attribute": data.get(
                                    "name", attr_key.replace("_", " ").title()
                                ),
                                "Statistic": "Max Value",
                                "Value": data.get("max_val", "N/A"),
                                "Player(s)": (
                                    ", ".join(data.get("max_players", []))
                                    if data.get("max_players")
                                    else ""
                                ),
                            }
                        )

                        export_data.append(
                            {
                                "Attribute": data.get(
                                    "name", attr_key.replace("_", " ").title()
                                ),
                                "Statistic": "Min Value",
                                "Value": data.get("min_val", "N/A"),
                                "Player(s)": (
                                    ", ".join(data.get("min_players", []))
                                    if data.get("min_players")
                                    else ""
                                ),
                            }
                        )
                    else:
                        export_data.append(
                            {
                                "Attribute": data.get(
                                    "name", attr_key.replace("_", " ").title()
                                ),
                                "Statistic": "Error",
                                "Value": data.get("error"),
                                "Player(s)": "",
                            }
                        )
            if export_data:
                df_export = pd.DataFrame(export_data)

        elif view_mode == "two_team_comparison":
            filename = f"comparison_{team1}_vs_{team2}_{comparison_scope}.csv"
            results_list = _get_two_team_comparison_data(
                conn, team1, team2, comparison_scope
            )
            if results_list:
                df = pd.DataFrame(results_list)
                df_export = df.rename(
                    columns={
                        "name": "Attribute",
                        "t1_value": f"{team1} Avg",
                        "t2_value": f"{team2} Avg",
                        "t1_better": f"{team1} > {team2}",
                        "t2_better": f"{team2} > {team1}",
                        "diff": "Abs Difference",
                    }
                )

        elif view_mode == "overview":
            filename = f"overview_stats_{scope}.csv"
            results_list, _ = _get_overview_stats(conn, scope)
            if results_list:

                df = pd.DataFrame(results_list)
                df_export = df.rename(
                    columns={
                        "nationality": "Nationality",
                        "player_count": f"Player Count ({scope.capitalize()})",
                        "avg_age": "Avg Age",
                        "avg_height": "Avg Height (cm)",
                        "avg_weight": "Avg Weight (kg)",
                        "avg_rating": "Avg Rating",
                    }
                )

        if not df_export.empty:
            csv_buffer = StringIO()
            df_export.to_csv(csv_buffer, index=False, encoding="utf-8")
            csv_buffer.seek(0)
            response = make_response(csv_buffer.getvalue())
            response.mimetype = "text/csv"
            response.headers["Content-Disposition"] = (
                f'attachment; filename="{filename}"'
            )
            return response
        else:
            logging.warning(
                f"No data generated for export view '{view_mode}' with current parameters."
            )

            return "Could not generate export data for the selected criteria.", 404

    except Exception as e:
        logging.exception("Error during CSV export generation:")
        return "An error occurred while generating the CSV export.", 500


@app.route("/players/export")
def player_comparison_export():
    conn = get_db()
    player1_name = request.args.get("player1")
    player2_name = request.args.get("player2")

    df_export = pd.DataFrame()
    filename_raw = "player_export.csv"
    filename_ascii = "player_export.csv"

    try:
        cursor = conn.cursor()
        player1_data = None
        player2_data = None

        if player1_name:
            cursor.execute(
                "SELECT * FROM players WHERE player_name = ?", (player1_name,)
            )
            p1_row = cursor.fetchone()
            if p1_row:
                player1_data = dict(p1_row)
            else:
                logging.warning(f"Export: Player 1 '{player1_name}' not found.")
                return "Player 1 not found.", 404

        if player2_name:
            cursor.execute(
                "SELECT * FROM players WHERE player_name = ?", (player2_name,)
            )
            p2_row = cursor.fetchone()
            if p2_row:
                player2_data = dict(p2_row)
            else:
                logging.warning(
                    f"Export: Player 2 '{player2_name}' not found. Exporting Player 1 only."
                )
                player2_name = None

        if player1_data and player2_data:

            p1_name_safe = (
                "".join(c for c in player1_name if ord(c) < 128)
                .replace(" ", "_")
                .replace(".", "")
                .replace("+", "")[:20]
            )
            p2_name_safe = (
                "".join(c for c in player2_name if ord(c) < 128)
                .replace(" ", "_")
                .replace(".", "")
                .replace("+", "")[:20]
            )
            filename_raw = f"comparison_{player1_name}_vs_{player2_name}.csv"
            filename_ascii = f"comparison_{p1_name_safe}_vs_{p2_name_safe}.csv"
            logging.info(
                f"Exporting comparison (no diff/better) for {player1_name} and {player2_name}"
            )

            comparison_export_list = []
            attributes_to_compare = [
                k
                for k in player1_data.keys()
                if k not in ["player_name", "nationality", "team_name"]
            ]
            for attr in attributes_to_compare:
                val1_raw = player1_data.get(attr)
                val2_raw = player2_data.get(attr)

                comp_entry = {
                    "Attribute": attr.replace("_", " ").title(),
                    f"{player1_name}": val1_raw,
                    f"{player2_name}": val2_raw,
                }
                comparison_export_list.append(comp_entry)

            if comparison_export_list:
                df_export = pd.DataFrame(comparison_export_list)

        elif player1_data:

            p1_name_safe = (
                "".join(c for c in player1_name if ord(c) < 128)
                .replace(" ", "_")
                .replace(".", "")
                .replace("+", "")[:30]
            )
            filename_raw = f"player_{player1_name}.csv"
            filename_ascii = f"player_{p1_name_safe}.csv"
            logging.info(f"Exporting data for single player: {player1_name}")
            single_export_list = [
                {"Attribute": key.replace("_", " ").title(), "Value": value}
                for key, value in player1_data.items()
            ]
            if single_export_list:
                df_export = pd.DataFrame(single_export_list)

        if not df_export.empty:
            csv_buffer = StringIO()
            df_export.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_buffer.seek(0)
            response = make_response(csv_buffer.getvalue())
            response.mimetype = "text/csv; charset=utf-8"

            try:
                filename_encoded = urllib.parse.quote(filename_raw, safe="")
                disposition = f"attachment; filename=\"{filename_ascii}\"; filename*=UTF-8''{filename_encoded}"
                response.headers["Content-Disposition"] = disposition
                logging.info(
                    f"CSV export '{filename_raw}' generated successfully with header: {disposition}"
                )
            except Exception as header_err:
                logging.error(
                    f"Error encoding filename for header: {header_err}. Using ASCII fallback."
                )
                response.headers["Content-Disposition"] = (
                    f'attachment; filename="{filename_ascii}"'
                )

            return response
        else:
            logging.warning(
                f"No data generated for player export with parameters: player1={player1_name}, player2={player2_name}"
            )
            return "Could not generate export data for the selected player(s).", 404

    except sqlite3.Error as e:
        logging.error(f"Database error during player export: {e}")
        return "Database error occurred during export.", 500
    except Exception as e:
        logging.exception("General error during player export:")
        return "An error occurred while generating the CSV export.", 500


@app.route("/attribute_distributions", methods=["GET"])
def attribute_distributions():
    conn = get_db()
    all_teams = []
    all_positions_for_dropdown = []
    available_attributes = []
    distribution_charts = []
    selected_players_for_table = []
    position_map_full_to_abbr = {}
    position_map_abbr_to_full = {}

    selected_team = request.args.get("team", "")
    selected_positions_full = request.args.getlist("position_filter")
    selected_attributes = request.args.getlist("attributes")
    scope = request.args.get("scope", "all")
    chart_type = request.args.get("chart_type", "histogram")

    logging.info(
        f"Request Params: team='{selected_team}', positions={selected_positions_full}, attributes={selected_attributes}, scope='{scope}', chart_type='{chart_type}'"
    )

    try:

        cursor_teams = conn.execute(
            "SELECT DISTINCT nationality FROM players ORDER BY nationality"
        )
        all_teams = [row["nationality"] for row in cursor_teams.fetchall()]
        available_attributes = _get_available_attributes()
        if not available_attributes:
            logging.error("Could not retrieve available attributes list.")
            return "Error: Could not load attribute data.", 500

        try:

            cursor_pos_roster = conn.execute(
                "SELECT DISTINCT assigned_position FROM rosters WHERE assigned_position IS NOT NULL AND assigned_position != '' ORDER BY assigned_position"
            )
            all_positions_for_dropdown = [
                row["assigned_position"] for row in cursor_pos_roster.fetchall()
            ]

            cursor_map = conn.execute(
                """
                SELECT DISTINCT p.primary_position, r.assigned_position
                FROM players p
                JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality
                WHERE p.primary_position IS NOT NULL AND p.primary_position != ''
                  AND r.assigned_position IS NOT NULL AND r.assigned_position != ''
            """
            )
            for row in cursor_map.fetchall():
                full_name = row["assigned_position"]
                abbr_name = row["primary_position"]
                if full_name and abbr_name:

                    if full_name not in position_map_full_to_abbr:
                        position_map_full_to_abbr[full_name] = abbr_name
                    if abbr_name not in position_map_abbr_to_full:
                        position_map_abbr_to_full[abbr_name] = full_name

            position_map_abbr_to_full["SS"] = "Second Striker"
            position_map_abbr_to_full["CF"] = "Centre Forward"
            position_map_abbr_to_full["GK"] = "Goalkeeper"
            position_map_abbr_to_full["CB"] = "Centre Back"
            position_map_abbr_to_full["LB"] = "Left Back"
            position_map_abbr_to_full["RB"] = "Right Back"
            position_map_abbr_to_full["DMF"] = "Defensive Midfielder"
            position_map_abbr_to_full["CMF"] = "Centre Midfielder"
            position_map_abbr_to_full["AMF"] = "Attacking Midfielder"
            position_map_abbr_to_full["LMF"] = "Left Midfielder"
            position_map_abbr_to_full["RMF"] = "Right Midfielder"
            position_map_abbr_to_full["LWF"] = "Left Wing Forward"
            position_map_abbr_to_full["RWF"] = "Right Wing Forward"

            for abbr, full in position_map_abbr_to_full.items():
                position_map_full_to_abbr[full] = abbr

            if not all_positions_for_dropdown:
                logging.warning(
                    "No assigned_position found in rosters, falling back to primary_position from players for dropdown."
                )
                cursor_pos_fallback = conn.execute(
                    "SELECT DISTINCT primary_position FROM players WHERE primary_position IS NOT NULL AND primary_position != '' ORDER BY primary_position"
                )
                fallback_positions = [
                    row["primary_position"] for row in cursor_pos_fallback.fetchall()
                ]

                all_positions_for_dropdown = sorted(
                    list(
                        set(
                            position_map_abbr_to_full.get(abbr, abbr)
                            for abbr in fallback_positions
                        )
                    )
                )

            if not all_positions_for_dropdown:
                logging.error("Failed to fetch any position names for filters.")

            logging.info(f"Fetched positions for filter: {all_positions_for_dropdown}")
            logging.info(
                f"Built position maps: Full->Abbr ({len(position_map_full_to_abbr)}), Abbr->Full ({len(position_map_abbr_to_full)})"
            )

            logging.info(
                f"Mapping check: 'Second Striker' -> '{position_map_full_to_abbr.get('Second Striker')}', 'Centre Forward' -> '{position_map_full_to_abbr.get('Centre Forward')}'"
            )
            logging.info(
                f"Mapping check: 'SS' -> '{position_map_abbr_to_full.get('SS')}', 'CF' -> '{position_map_abbr_to_full.get('CF')}'"
            )

        except sqlite3.Error as e:
            logging.error(
                f"Database error fetching positions/building maps: {e}. Using empty list.",
                exc_info=True,
            )
            all_positions_for_dropdown = []
            position_map_full_to_abbr = {}
            position_map_abbr_to_full = {}

        player_query = ""
        player_params = []
        selected_players_data = []

        attributes_to_select = set(["player_name", "nationality"])
        if selected_attributes:
            attributes_to_select.update(selected_attributes)

        cursor_info = conn.execute("PRAGMA table_info(players)")
        db_player_cols = {info[1] for info in cursor_info.fetchall()}

        if scope == "all":
            attributes_to_select.add("primary_position")
        elif scope == "squad":
            attributes_to_select.add("assigned_position")

        valid_attributes_to_select_players = [
            attr
            for attr in attributes_to_select
            if attr in db_player_cols or attr == "assigned_position"
        ]

        if scope == "squad":

            player_cols_select = [
                f'p."{attr}"'
                for attr in valid_attributes_to_select_players
                if attr in db_player_cols
            ]
            roster_cols_select = []
            if "assigned_position" in valid_attributes_to_select_players:
                roster_cols_select.append("r.assigned_position AS position")

            if 'p."player_name"' not in player_cols_select:
                player_cols_select.append('p."player_name"')
            if 'p."nationality"' not in player_cols_select:
                player_cols_select.append('p."nationality"')

            select_clause = ", ".join(
                list(set(player_cols_select + roster_cols_select))
            )

            player_query = f"SELECT {select_clause} FROM players p JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality WHERE 1=1"
            player_params = []
            if selected_team:
                player_query += " AND p.nationality = ?"
                player_params.append(selected_team)
            if selected_positions_full:

                placeholders = ",".join("?" for _ in selected_positions_full)
                player_query += f" AND r.assigned_position IN ({placeholders})"
                player_params.extend(selected_positions_full)

        elif scope == "all":

            player_cols_select = [
                f'p."{attr}"'
                for attr in valid_attributes_to_select_players
                if attr in db_player_cols
            ]

            if 'p."player_name"' not in player_cols_select:
                player_cols_select.append('p."player_name"')
            if 'p."nationality"' not in player_cols_select:
                player_cols_select.append('p."nationality"')
            if (
                'p."primary_position"' not in player_cols_select
                and "primary_position" in valid_attributes_to_select_players
            ):
                player_cols_select.append('p."primary_position"')

            select_clause = ", ".join(list(set(player_cols_select)))

            player_query = f"SELECT {select_clause} FROM players p WHERE 1=1"
            player_params = []
            if selected_team:
                player_query += " AND p.nationality = ?"
                player_params.append(selected_team)

            if selected_positions_full:
                selected_positions_abbr = []
                for full_name in selected_positions_full:
                    abbr = position_map_full_to_abbr.get(full_name)
                    if abbr:
                        selected_positions_abbr.append(abbr)
                    else:
                        logging.warning(
                            f"Could not map position '{full_name}' to an abbreviation for filtering."
                        )

                if selected_positions_abbr:
                    placeholders = ",".join("?" for _ in selected_positions_abbr)
                    player_query += f" AND p.primary_position IN ({placeholders})"
                    player_params.extend(selected_positions_abbr)
                    logging.info(
                        f"Applying position filter using abbreviations: {selected_positions_abbr}"
                    )
                else:
                    logging.warning(
                        f"Could not translate any selected positions {selected_positions_full} to abbreviations. Position filter ignored."
                    )

        if player_query:
            logging.info(f"Executing player data query: {player_query}")
            logging.info(f"With parameters: {player_params}")
            try:
                cursor_players = conn.execute(player_query, player_params)
                selected_players_data = [dict(row) for row in cursor_players.fetchall()]
                logging.info(f"Fetched {len(selected_players_data)} players raw.")

                selected_players_for_table = []
                for player_raw in selected_players_data:
                    player_processed = player_raw.copy()
                    if scope == "all":
                        abbr = player_processed.get("primary_position")
                        full_name = position_map_abbr_to_full.get(abbr, abbr)
                        player_processed["position"] = full_name
                    elif "position" not in player_processed:
                        player_processed["position"] = player_processed.get(
                            "assigned_position", "N/A"
                        )

                    selected_players_for_table.append(player_processed)
                logging.info(
                    f"Processed {len(selected_players_for_table)} players for table display."
                )

            except sqlite3.Error as db_err:
                logging.error(
                    f"Database error executing player query: {db_err}", exc_info=True
                )
                selected_players_data = []
                selected_players_for_table = []
            except Exception as e:
                logging.error(
                    f"Error fetching or processing player data: {e}", exc_info=True
                )
                selected_players_data = []
                selected_players_for_table = []
        else:
            logging.info("No player data query executed.")

        if selected_attributes and selected_players_data:
            logging.info(
                f"Generating charts for {len(selected_attributes)} attributes and {len(selected_players_data)} players."
            )
            df_players = pd.DataFrame(selected_players_data)

            for attribute in selected_attributes:
                if attribute in df_players.columns:

                    numeric_values = pd.to_numeric(
                        df_players[attribute], errors="coerce"
                    ).dropna()

                    if not numeric_values.empty:
                        try:
                            fig = go.Figure()
                            attr_display_name = attribute.replace("_", " ").title()
                            chart_title = f"Distribuce: {attr_display_name}"
                            if selected_team:
                                chart_title += f" ({selected_team})"

                            if selected_positions_full:
                                pos_display = ", ".join(selected_positions_full)
                                if len(pos_display) > 50:
                                    pos_display = pos_display[:47] + "..."
                                chart_title += f" | Pozice: {pos_display}"

                            if chart_type == "box":
                                fig.add_trace(
                                    go.Box(y=numeric_values, name=attr_display_name)
                                )
                                fig.update_layout(
                                    title=chart_title, yaxis_title=attr_display_name
                                )
                            else:
                                fig.add_trace(
                                    go.Histogram(
                                        x=numeric_values, name=attr_display_name
                                    )
                                )
                                fig.update_layout(
                                    title=chart_title,
                                    xaxis_title=attr_display_name,
                                    yaxis_title="Počet hráčů",
                                )

                            chart_json = json.dumps(
                                fig, cls=plotly.utils.PlotlyJSONEncoder
                            )
                            distribution_charts.append(
                                {
                                    "attribute": attribute,
                                    "title_attribute": attr_display_name,
                                    "json": chart_json,
                                }
                            )
                            logging.info(
                                f"Successfully created chart for {attribute}, JSON length: {len(chart_json)}"
                            )
                        except Exception as chart_err:
                            logging.error(
                                f"Error generating chart for attribute '{attribute}': {chart_err}",
                                exc_info=True,
                            )
                            distribution_charts.append(
                                {
                                    "attribute": attribute,
                                    "json": "{}",
                                    "error": str(chart_err),
                                }
                            )
                    else:
                        logging.warning(
                            f"No valid numeric data for attribute '{attribute}' after filtering. Skipping chart."
                        )
                        distribution_charts.append(
                            {
                                "attribute": attribute,
                                "json": "{}",
                                "error": "No numeric data",
                            }
                        )
                else:
                    logging.warning(
                        f"Selected attribute '{attribute}' not found in fetched player data columns: {df_players.columns}. Skipping chart."
                    )
                    distribution_charts.append(
                        {
                            "attribute": attribute,
                            "json": "{}",
                            "error": "Attribute not found in data",
                        }
                    )
        elif not selected_players_data and (selected_team or selected_positions_full):
            logging.info(
                "No players found matching the specified filters. No charts generated."
            )

    except sqlite3.Error as e:
        logging.error(
            f"Database error encountered in /attribute_distributions route: {e}",
            exc_info=True,
        )
        return f"Database error occurred: {str(e)}", 500
    except Exception as e:
        logging.error(
            f"An unexpected general error occurred in /attribute_distributions: {e}",
            exc_info=True,
        )
        return f"An unexpected error occurred: {str(e)}", 500

    attributes_by_group_for_template = {}
    if available_attributes:
        for group, attrs in ATTRIBUTE_GROUPS.items():
            group_attrs_available = sorted(
                [attr for attr in attrs if attr in available_attributes]
            )
            if group_attrs_available:
                attributes_by_group_for_template[group] = group_attrs_available
        grouped_attrs_set = {
            attr
            for group_list in attributes_by_group_for_template.values()
            for attr in group_list
        }
        ungrouped_available = sorted(
            [attr for attr in available_attributes if attr not in grouped_attrs_set]
        )
        if ungrouped_available:
            attributes_by_group_for_template.setdefault("Other", []).extend(
                ungrouped_available
            )
            attributes_by_group_for_template["Other"].sort()
    else:
        logging.warning(
            "Available attributes list is empty, cannot create grouped attributes for template."
        )

    selected_attributes_display = {}
    if "db_player_cols" in locals():
        selected_attributes_display = {
            attr: attr.replace("_", " ").title()
            for attr in selected_attributes
            if attr in db_player_cols
        }
    else:
        logging.warning(
            "db_player_cols not defined, cannot create selected_attributes_display."
        )

    logging.info(
        f"Rendering template with {len(selected_players_for_table)} players in selected_players list."
    )
    return render_template(
        "attribute_distributions.html",
        title="Distribuce Atributů",
        all_teams=all_teams,
        all_positions=all_positions_for_dropdown,
        position_categories=position_categories,
        available_attributes=available_attributes,
        selected_team=selected_team,
        selected_positions=selected_positions_full,
        selected_attributes=selected_attributes,
        selected_attributes_display=selected_attributes_display,
        scope=scope,
        chart_type=chart_type,
        distribution_charts=distribution_charts,
        selected_players=selected_players_for_table,
        attributes_by_group=attributes_by_group_for_template,
        active_page="distributions",
        get_flag_emoji=get_flag_emoji,
    )


@app.context_processor
def utility_processor():
    return dict(get_flag_emoji=get_flag_emoji)


@app.errorhandler(404)
def not_found_error(error):

    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):

    logging.exception("Internal Server Error:")
    return render_template("500.html"), 500


@app.route("/analysis/correlation", methods=["GET"])
def correlation_analysis():
    conn = get_db()
    all_teams = []
    all_positions_for_dropdown = []
    available_attributes = []
    correlation_matrix_json = None
    selected_players_for_table = []
    position_map_full_to_abbr = {}
    position_map_abbr_to_full = {}
    attr_stats = {}

    position_categories = {
        "Attacker": [
            "CF",
            "SS",
            "LWF",
            "RWF",
            "Centre Forward",
            "Second Striker",
            "Left Wing Forward",
            "Right Wing Forward",
        ],
        "Midfielder": [
            "AMF",
            "CMF",
            "DMF",
            "LMF",
            "RMF",
            "Attacking Midfielder",
            "Centre Midfielder",
            "Defensive Midfielder",
            "Left Midfielder",
            "Right Midfielder",
        ],
        "Defender": ["CB", "LB", "RB", "Centre Back", "Left Back", "Right Back"],
        "Goalkeeper": ["GK", "Goalkeeper"],
    }

    attribute_groups = {
        "Physical": [
            "height",
            "weight",
            "age",
            "speed",
            "acceleration",
            "jumping",
            "physical_contact",
            "balance",
            "stamina",
        ],
        "Attacking": ["offensive_awareness", "finishing", "kicking_power", "heading"],
        "Ball Control": ["ball_control", "dribbling", "tight_possession"],
        "Passing": ["low_pass", "lofted_pass", "curl", "set_piece_taking"],
        "Defending": [
            "defensive_awareness",
            "tackling",
            "aggression",
            "defensive_engagement",
        ],
        "Goalkeeping": [
            "gk_awareness",
            "gk_catching",
            "gk_parrying",
            "gk_reflexes",
            "gk_reach",
        ],
    }

    selected_team = request.args.get("team", "")
    selected_positions_full = request.args.getlist("position_filter")
    selected_position_categories = request.args.getlist("position_category")
    selected_attributes = request.args.getlist("attributes")
    scope = request.args.get("scope", "all")
    chart_type = request.args.get("chart_type", "heatmap")
    x_attribute = request.args.get("x_attribute", "")
    y_attribute = request.args.get("y_attribute", "")

    logging.info(
        f"Correlation Analysis Request: team='{selected_team}', positions={selected_positions_full}, "
        f"position_categories={selected_position_categories}, "
        f"attributes={selected_attributes}, scope='{scope}', chart_type='{chart_type}', "
        f"x={x_attribute}, y={y_attribute}"
    )

    try:

        cursor_teams = conn.execute(
            "SELECT DISTINCT nationality FROM players ORDER BY nationality"
        )
        all_teams = [row["nationality"] for row in cursor_teams.fetchall()]
        available_attributes = _get_available_attributes()

        if not available_attributes:
            logging.error("Could not retrieve available attributes list.")
            return "Error: Could not load available player attributes.", 500

        try:
            cursor_pos_roster = conn.execute(
                "SELECT DISTINCT assigned_position FROM rosters WHERE assigned_position IS NOT NULL AND assigned_position != '' ORDER BY assigned_position"
            )
            all_positions_for_dropdown = [
                row["assigned_position"] for row in cursor_pos_roster.fetchall()
            ]

            cursor_map = conn.execute(
                """
                SELECT DISTINCT p.primary_position, r.assigned_position
                FROM players p
                JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality
                WHERE p.primary_position IS NOT NULL AND r.assigned_position IS NOT NULL
            """
            )
            for row in cursor_map.fetchall():
                full_name = row["assigned_position"]
                abbr_name = row["primary_position"]
                if full_name and abbr_name:
                    position_map_full_to_abbr[full_name] = abbr_name
                    position_map_abbr_to_full[abbr_name] = full_name

            position_map_abbr_to_full["RWF"] = "Right Wing Forward"
            position_map_abbr_to_full["LWF"] = "Left Wing Forward"
            position_map_abbr_to_full["RB"] = "Right Back"
            position_map_abbr_to_full["LB"] = "Left Back"
            position_map_abbr_to_full["CB"] = "Centre Back"
            position_map_abbr_to_full["CF"] = "Centre Forward"
            position_map_abbr_to_full["SS"] = "Second Striker"
            position_map_abbr_to_full["GK"] = "Goalkeeper"
            position_map_abbr_to_full["DMF"] = "Defensive Midfielder"
            position_map_abbr_to_full["CMF"] = "Centre Midfielder"
            position_map_abbr_to_full["AMF"] = "Attacking Midfielder"
            position_map_abbr_to_full["LMF"] = "Left Midfielder"
            position_map_abbr_to_full["RMF"] = "Right Midfielder"

            for abbr, full in position_map_abbr_to_full.items():
                position_map_full_to_abbr[full] = abbr

            if not all_positions_for_dropdown:
                logging.warning(
                    "No positions found in 'rosters', falling back to 'players' primary_position for dropdown."
                )
                cursor_pos_players = conn.execute(
                    "SELECT DISTINCT primary_position FROM players WHERE primary_position IS NOT NULL AND primary_position != '' ORDER BY primary_position"
                )
                all_positions_for_dropdown = [
                    row["primary_position"] for row in cursor_pos_players.fetchall()
                ]

        except sqlite3.Error as e:
            logging.error(
                f"Database error fetching positions/building map: {e}. Using fallback.",
                exc_info=True,
            )
            try:
                cursor_pos_players = conn.execute(
                    "SELECT DISTINCT primary_position FROM players WHERE primary_position IS NOT NULL AND primary_position != '' ORDER BY primary_position"
                )
                all_positions_for_dropdown = [
                    row["primary_position"] for row in cursor_pos_players.fetchall()
                ]
            except Exception as fallback_e:
                logging.error(
                    f"Error during position fallback query: {fallback_e}", exc_info=True
                )
                all_positions_for_dropdown = []
            position_map_full_to_abbr = {}
            position_map_abbr_to_full = {}
        player_query = ""
        player_params = []
        selected_players_data = []

        expanded_position_filters = []

        if selected_positions_full:
            expanded_position_filters.extend(selected_positions_full)

        for category in selected_position_categories:
            if category in position_categories:
                expanded_position_filters.extend(position_categories[category])

        expanded_position_filters = list(set(expanded_position_filters))

        attributes_to_select = set(
            ["player_name", "overall_rating", "primary_position", "nationality"]
        )
        required_logic_cols = ["assigned_position"]

        if selected_attributes:
            attributes_to_select.update(selected_attributes)
        elif chart_type == "heatmap":
            attributes_to_select.update(available_attributes)

        if chart_type == "scatter" and x_attribute and y_attribute:
            attributes_to_select.add(x_attribute)
            attributes_to_select.add(y_attribute)

        valid_attributes_to_select = [
            attr
            for attr in attributes_to_select
            if attr in available_attributes
            or attr
            in [
                "player_name",
                "overall_rating",
                "primary_position",
                "assigned_position",
                "nationality",
            ]
        ]

        for req_col in required_logic_cols:
            if req_col not in valid_attributes_to_select and (
                req_col in available_attributes or req_col in ["assigned_position"]
            ):
                valid_attributes_to_select.append(req_col)

        valid_attributes_to_select = list(set(valid_attributes_to_select))

        if scope == "squad":
            player_cols = [
                f"p.{attr}"
                for attr in valid_attributes_to_select
                if attr not in ["assigned_position"]
            ]
            roster_cols = []
            if "assigned_position" in valid_attributes_to_select:
                roster_cols.append("r.assigned_position")
            if "p.nationality" not in player_cols:
                player_cols.append("p.nationality")

            select_clause = ", ".join(list(set(player_cols + roster_cols)))

            if selected_team:
                player_query = f"""SELECT {select_clause} FROM players p 
                                  JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality 
                                  WHERE r.nationality = ? """
                player_params = [selected_team]

                if expanded_position_filters:
                    placeholders = ",".join("?" for _ in expanded_position_filters)
                    player_query += f" AND r.assigned_position IN ({placeholders})"
                    player_params.extend(expanded_position_filters)
            else:
                player_query = f"""SELECT {select_clause} FROM players p 
                                  JOIN rosters r ON p.player_name = r.player_name AND p.nationality = r.nationality """
                player_params = []

                if expanded_position_filters:
                    placeholders = ",".join("?" for _ in expanded_position_filters)
                    player_query += f" WHERE r.assigned_position IN ({placeholders})"
                    player_params.extend(expanded_position_filters)
        elif scope == "all":
            player_cols = [
                f"p.{attr}"
                for attr in valid_attributes_to_select
                if attr != "assigned_position"
            ]

            if (
                "p.primary_position" not in player_cols
                and "primary_position" in valid_attributes_to_select
            ):
                player_cols.append("p.primary_position")
            if (
                "p.nationality" not in player_cols
                and "nationality" in valid_attributes_to_select
            ):
                player_cols.append("p.nationality")

            select_clause = ", ".join(list(set(player_cols)))
            player_query = f"SELECT {select_clause} FROM players p WHERE 1=1"
            player_params = []

            if selected_team:
                player_query += " AND p.nationality = ?"
                player_params.append(selected_team)

            position_abbrs = []
            if expanded_position_filters:
                for pos in expanded_position_filters:
                    if pos in position_map_full_to_abbr:
                        position_abbrs.append(position_map_full_to_abbr[pos])
                    else:
                        position_abbrs.append(pos)

                if position_abbrs:
                    placeholders = ",".join("?" for _ in position_abbrs)
                    player_query += f" AND p.primary_position IN ({placeholders})"
                    player_params.extend(position_abbrs)

        if player_query:
            logging.info(f"Executing player data query: {player_query}")
            logging.info(f"With parameters: {player_params}")

            try:
                cursor_players = conn.execute(player_query, player_params)
                raw_players_data = [dict(row) for row in cursor_players.fetchall()]
                logging.info(f"Fetched {len(raw_players_data)} players.")

                selected_players_data = []
                selected_players_for_table = []

                for player_dict in raw_players_data:
                    processed_player = dict(player_dict)

                    if processed_player.get("assigned_position"):
                        processed_player["position"] = processed_player[
                            "assigned_position"
                        ]
                    elif (
                        processed_player.get("primary_position")
                        and processed_player["primary_position"]
                        in position_map_abbr_to_full
                    ):
                        processed_player["position"] = position_map_abbr_to_full[
                            processed_player["primary_position"]
                        ]
                    else:
                        processed_player["position"] = processed_player.get(
                            "primary_position", "Unknown"
                        )

                    selected_players_data.append(processed_player)

                    table_row = {
                        "player_name": processed_player.get("player_name", "N/A"),
                        "position": processed_player.get("position", "N/A"),
                        "nationality": processed_player.get("nationality", "N/A"),
                    }

                    for attr_key in available_attributes:
                        if attr_key in processed_player:
                            try:
                                val = processed_player[attr_key]
                                if val is not None:
                                    numeric_val = float(val)
                                    if numeric_val == int(numeric_val):
                                        table_row[attr_key] = int(numeric_val)
                                    else:
                                        table_row[attr_key] = round(numeric_val, 1)
                                else:
                                    table_row[attr_key] = "N/A"
                            except (ValueError, TypeError):
                                table_row[attr_key] = processed_player.get(
                                    attr_key, "N/A"
                                )

                    selected_players_for_table.append(table_row)

                selected_players_for_table.sort(
                    key=lambda x: (x.get("nationality", ""), x.get("player_name", ""))
                )

            except sqlite3.Error as db_err:
                logging.error(
                    f"Database error fetching player list: {db_err}", exc_info=True
                )
                selected_players_data = []
                selected_players_for_table = []
            except Exception as e:
                logging.error(f"Error processing player list: {e}", exc_info=True)
                selected_players_data = []
                selected_players_for_table = []
        else:
            logging.info("No player data query executed.")

        attrs_for_correlation = selected_attributes

        if chart_type == "heatmap" and not attrs_for_correlation:
            logging.info(
                "No attributes selected for correlation matrix, using all available attributes"
            )
            attrs_for_correlation = available_attributes

        if selected_players_data and (
            (chart_type == "heatmap" and len(attrs_for_correlation) >= 2)
            or (chart_type == "scatter" and x_attribute and y_attribute)
        ):

            numeric_data = {}

            for attr in attrs_for_correlation:
                attr_values = []
                for player in selected_players_data:
                    try:
                        val = player.get(attr)
                        if val is not None:
                            numeric_val = float(val)
                            if not pd.isna(numeric_val):
                                attr_values.append(numeric_val)
                            else:
                                attr_values.append(None)
                        else:
                            attr_values.append(None)
                    except (ValueError, TypeError):
                        attr_values.append(None)

                if attr_values and sum(1 for v in attr_values if v is not None) >= 2:
                    numeric_data[attr] = attr_values

                    valid_values = [v for v in attr_values if v is not None]
                    if valid_values:
                        attr_stats[attr] = {
                            "mean": round(sum(valid_values) / len(valid_values), 1),
                            "min": min(valid_values),
                            "max": max(valid_values),
                            "count": len(valid_values),
                        }

            df = pd.DataFrame(numeric_data)

            title_prefix = f"{selected_team} - " if selected_team else ""
            pos_suffix = (
                f" ({', '.join(expanded_position_filters)})"
                if expanded_position_filters
                else ""
            )
            scope_detail = ""

            if scope == "squad":
                scope_detail = (
                    "Selected Team Roster Players"
                    if selected_team
                    else "All Teams - Roster Players"
                )
            else:
                scope_detail = "All Players"
                if selected_team:
                    scope_detail += " (Filtered by Team)"

            position_filter_detail = ""
            if expanded_position_filters:
                position_filter_detail = f"(Filtered by Position)"

            full_scope_description = f"{scope_detail} {position_filter_detail}".strip()

            if chart_type == "heatmap" and not df.empty and df.shape[1] >= 2:
                try:
                    corr_matrix = df.corr(method="pearson", numeric_only=True).round(2)

                    logging.info(
                        f"Generated correlation matrix with shape {corr_matrix.shape}"
                    )

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr_matrix.values.tolist(),
                            x=corr_matrix.columns.tolist(),
                            y=corr_matrix.columns.tolist(),
                            colorscale="RdBu_r",
                            zmid=0,
                            text=corr_matrix.values.round(2).tolist(),
                            hovertemplate="<b>X</b>: %{x}<br>"
                            + "<b>Y</b>: %{y}<br>"
                            + "<b>Correlation</b>: %{text}<extra></extra>",
                            hoverongaps=False,
                        )
                    )

                    fig.update_layout(
                        title=f"{title_prefix}Correlation Matrix{pos_suffix}<br><sup>Scope: {full_scope_description}</sup>",
                        height=800,
                        width=900,
                        template="plotly_white",
                        title_font_size=14,
                        xaxis_title="Attributes",
                        yaxis_title="Attributes",
                    )

                    try:
                        correlation_matrix_json = json.dumps(
                            fig, cls=plotly.utils.PlotlyJSONEncoder
                        )
                        logging.info(
                            f"Successfully serialized correlation matrix to JSON (length: {len(correlation_matrix_json)})"
                        )
                    except Exception as json_err:
                        logging.error(
                            f"Error serializing correlation matrix to JSON: {json_err}"
                        )
                        correlation_matrix_json = None

                except Exception as e:
                    logging.error(
                        f"Error creating correlation heatmap: {e}", exc_info=True
                    )
                    correlation_matrix_json = None

            elif chart_type == "scatter" and x_attribute and y_attribute:
                try:
                    scatter_data = []
                    for player in selected_players_data:
                        try:
                            x_val = float(player.get(x_attribute, "NaN"))
                            y_val = float(player.get(y_attribute, "NaN"))
                            if not (pd.isna(x_val) or pd.isna(y_val)):
                                scatter_data.append(
                                    {
                                        "x": x_val,
                                        "y": y_val,
                                        "player": player.get("player_name", "Unknown"),
                                        "nationality": player.get(
                                            "nationality", "Unknown"
                                        ),
                                        "position": player.get("position", "Unknown"),
                                    }
                                )
                        except (ValueError, TypeError):
                            continue

                    if scatter_data:
                        x_values = [p["x"] for p in scatter_data]
                        y_values = [p["y"] for p in scatter_data]
                        corr_coef = round(np.corrcoef(x_values, y_values)[0, 1], 2)

                        fig = go.Figure(
                            data=go.Scatter(
                                x=[p["x"] for p in scatter_data],
                                y=[p["y"] for p in scatter_data],
                                mode="markers",
                                marker=dict(
                                    size=10, color="rgb(55, 83, 109)", opacity=0.7
                                ),
                                text=[
                                    f"Player: {p['player']}<br>Nationality: {p['nationality']}<br>Position: {p['position']}"
                                    for p in scatter_data
                                ],
                                hoverinfo="text",
                            )
                        )

                        z = np.polyfit(x_values, y_values, 1)
                        line_x = [min(x_values), max(x_values)]
                        line_y = [z[0] * x + z[1] for x in line_x]

                        fig.add_trace(
                            go.Scatter(
                                x=line_x,
                                y=line_y,
                                mode="lines",
                                line=dict(color="red", width=2),
                                name=f"Trend (r = {corr_coef})",
                            )
                        )

                        x_title = x_attribute.replace("_", " ").title()
                        y_title = y_attribute.replace("_", " ").title()

                        fig.update_layout(
                            title=f"{title_prefix}Relationship: {x_title} vs {y_title}{pos_suffix}<br>"
                            f"<sup>Scope: {full_scope_description} | Correlation: {corr_coef}</sup>",
                            xaxis_title=x_title,
                            yaxis_title=y_title,
                            height=600,
                            template="plotly_white",
                            showlegend=True,
                        )

                        correlation_matrix_json = json.dumps(
                            fig, cls=plotly.utils.PlotlyJSONEncoder
                        )
                        logging.info(
                            f"Generated scatter plot for {x_attribute} vs {y_attribute}"
                        )

                    else:
                        logging.warning(
                            f"No valid data points for scatter plot {x_attribute} vs {y_attribute}"
                        )
                        correlation_matrix_json = None

                except Exception as e:
                    logging.error(f"Error creating scatter plot: {e}", exc_info=True)
                    correlation_matrix_json = None

    except sqlite3.Error as e:
        logging.error(f"Database error in correlation analysis: {e}", exc_info=True)
        return f"Database error occurred: {str(e)}", 500
    except Exception as e:
        logging.error(f"General error in correlation analysis: {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}", 500

    selected_attributes_display = {
        attr_key: attr_key.replace("_", " ").title()
        for attr_key in attrs_for_correlation
    }

    attributes_by_group = {}
    for group, attrs in attribute_groups.items():
        group_attrs = [attr for attr in attrs if attr in available_attributes]
        if group_attrs:
            attributes_by_group[group] = group_attrs

    ungrouped_attrs = [
        attr
        for attr in available_attributes
        if not any(attr in group_attrs for group_attrs in attribute_groups.values())
    ]
    if ungrouped_attrs:
        attributes_by_group["Other"] = ungrouped_attrs

    logging.info(
        f"Passing to template: {len(selected_players_for_table)} players, attributes_display={selected_attributes_display}"
    )
    if not selected_players_for_table:
        logging.warning("No players data to display in table")

    return render_template(
        "correlation_analysis.html",
        title="Korelační Analýza",
        all_teams=all_teams,
        all_positions=all_positions_for_dropdown,
        position_categories=position_categories,
        selected_position_categories=selected_position_categories,
        available_attributes=available_attributes,
        attributes_by_group=attributes_by_group,
        selected_team=selected_team,
        selected_positions=selected_positions_full,
        selected_attributes=selected_attributes,
        selected_attributes_display=selected_attributes_display,
        x_attribute=x_attribute,
        y_attribute=y_attribute,
        scope=scope,
        chart_type=chart_type,
        correlation_matrix_json=correlation_matrix_json,
        selected_players=selected_players_for_table,
        attr_stats=attr_stats,
        get_flag_emoji=get_flag_emoji,
    )
