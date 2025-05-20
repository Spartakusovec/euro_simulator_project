import sqlite3
import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATABASE_PATH = os.path.join(BASE_DIR, "data", "database.db")
PLAYERS_CSV_PATH = os.path.join(
    BASE_DIR, "data", "info_with_free_agents_2025_03_W2.csv"
)
MATCHES_CSV_PATH = os.path.join(BASE_DIR, "data", "vsechny_evropske_zapasy.csv")


def create_connection(db_file):
    conn = None
    try:
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        conn = sqlite3.connect(db_file)
        logging.info(f"SQLite version: {sqlite3.sqlite_version}")
        logging.info(f"Úspěšně připojeno k {db_file}")
    except sqlite3.Error as e:
        logging.error(f"Chyba při připojování k databázi: {e}")
    return conn


def create_tables(conn):
    if conn is None:
        logging.error("Nelze vytvořit tabulky: Chybí připojení.")
        return

    sql_create_players_table = """CREATE TABLE IF NOT EXISTS players (...);"""
    sql_create_teams_table = (
        "CREATE TABLE IF NOT EXISTS teams (nationality TEXT PRIMARY KEY NOT NULL);"
    )
    sql_create_historical_matches_table = (
        """CREATE TABLE IF NOT EXISTS historical_matches (...);"""
    )
    sql_create_rosters_table = """CREATE TABLE IF NOT EXISTS rosters (...);"""
    sql_create_simulation_probabilities_table = (
        """CREATE TABLE IF NOT EXISTS simulation_probabilities (...);"""
    )
    sql_create_simulation_run_details_table = (
        """CREATE TABLE IF NOT EXISTS simulation_run_details (...);"""
    )

    sql_create_elo_snapshots_table = """
    CREATE TABLE IF NOT EXISTS elo_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        simulation_id INTEGER DEFAULT 1 NOT NULL,
        match_order INTEGER NOT NULL,
        stage TEXT,
        match_description TEXT,
        nationality TEXT NOT NULL,
        elo_after_match REAL NOT NULL
        -- Odebráno: FOREIGN KEY (nationality) REFERENCES teams (nationality)
    );
    """

    try:
        cursor = conn.cursor()
        logging.info("Vytváření/kontrola tabulek...")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS players (player_name TEXT, team_name TEXT, nationality TEXT, height INTEGER, weight INTEGER, age INTEGER, foot TEXT, rating TEXT, primary_position TEXT, secondary_positions TEXT, backup_positions TEXT, overall_rating INTEGER, league TEXT, region TEXT, offensive_awareness INTEGER, ball_control INTEGER, dribbling INTEGER, tight_possession INTEGER, low_pass INTEGER, lofted_pass INTEGER, finishing INTEGER, heading INTEGER, set_piece_taking INTEGER, curl INTEGER, defensive_awareness INTEGER, tackling INTEGER, aggression INTEGER, defensive_engagement INTEGER, speed INTEGER, acceleration INTEGER, kicking_power INTEGER, jumping INTEGER, physical_contact INTEGER, balance INTEGER, stamina INTEGER, gk_awareness INTEGER, gk_catching INTEGER, gk_parrying INTEGER, gk_reflexes INTEGER, gk_reach INTEGER, weak_foot_usage TEXT, weak_foot_accuracy TEXT, form TEXT, injury_resistance TEXT, UNIQUE(player_name, height, weight, age, nationality) );"
        )
        logging.info("Tabulka 'players' zkontrolována/vytvořena.")
        cursor.execute(sql_create_teams_table)
        logging.info("Tabulka 'teams' zkontrolována/vytvořena.")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS historical_matches ( match_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, team_a TEXT, team_b TEXT, score_a INTEGER, score_b INTEGER, tournament TEXT, location TEXT, rating_change_a REAL, rating_change_b REAL, new_rating_a REAL, new_rating_b REAL );"
        )
        logging.info("Tabulka 'historical_matches' zkontrolována/vytvořena.")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS rosters ( roster_id INTEGER PRIMARY KEY AUTOINCREMENT, nationality TEXT NOT NULL, player_name TEXT NOT NULL, assigned_position TEXT NOT NULL, overall_rating_in_position INTEGER NOT NULL, formation_name TEXT NOT NULL, FOREIGN KEY (nationality) REFERENCES teams (nationality) );"
        )
        logging.info("Tabulka 'rosters' zkontrolována/vytvořena.")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS simulation_probabilities ( nationality TEXT PRIMARY KEY NOT NULL, win_prob REAL DEFAULT 0.0, final_prob REAL DEFAULT 0.0, semi_prob REAL DEFAULT 0.0, quarter_prob REAL DEFAULT 0.0, FOREIGN KEY (nationality) REFERENCES teams (nationality) );"
        )
        logging.info("Tabulka 'simulation_probabilities' zkontrolována/vytvořena.")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS simulation_run_details ( detail_id INTEGER PRIMARY KEY AUTOINCREMENT, simulation_id INTEGER DEFAULT 1, stage TEXT NOT NULL, description TEXT NOT NULL );"
        )
        logging.info("Tabulka 'simulation_run_details' zkontrolována/vytvořena.")
        cursor.execute(sql_create_elo_snapshots_table)
        logging.info("Tabulka 'elo_snapshots' zkontrolována/vytvořena (bez FK).")
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Chyba při vytváření tabulek: {e}")


def sanitize_column_names(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)
    return df


def populate_players(conn, csv_path):
    if conn is None:
        logging.error("...")
        return None
    try:
        logging.info(f"Načítání dat hráčů z {csv_path}...")
        df = pd.read_csv(csv_path)
        logging.info(f"Načteno {len(df)} záznamů hráčů.")
        df = sanitize_column_names(df)
        df.rename(columns={"primary_position": "primary_position"}, inplace=True)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(players)")
        db_columns = [info[1] for info in cursor.fetchall()]
        df_columns = df.columns.tolist()
        columns_to_insert = [col for col in df_columns if col in db_columns]
        missing_cols = [col for col in db_columns if col not in df_columns]
        extra_cols = [col for col in df_columns if col not in db_columns]
        if missing_cols:
            logging.warning(f"... {missing_cols}")
        if extra_cols:
            logging.warning(f"... {extra_cols}")
        df_to_insert = df[columns_to_insert]
        logging.info(
            "Vkládání dat hráčů do tabulky 'players' (nahrazení stávajících)..."
        )
        df_to_insert.to_sql("players", conn, if_exists="replace", index=False)
        logging.info(f"Tabulka 'players' úspěšně naplněna {len(df_to_insert)} záznamy.")
        return df_to_insert
    except Exception as e:
        logging.error(f"Chyba při plnění 'players': {e}")
        logging.exception("Detail:")
        return None


def populate_teams(conn, players_df):
    if conn is None or players_df is None:
        logging.error("...")
        return
    try:
        logging.info("Extrahování unikátních národností...")
        unique_teams = players_df["nationality"].unique()
        teams_df = pd.DataFrame(unique_teams, columns=["nationality"])
        logging.info("Vkládání unikátních národností do tabulky 'teams'...")
        teams_df.to_sql("teams", conn, if_exists="replace", index=False)
        logging.info(f"Tabulka 'teams' úspěšně naplněna {len(teams_df)} týmy.")
    except Exception as e:
        logging.error(f"Chyba při plnění 'teams': {e}")


def populate_historical_matches(conn, csv_path):
    if conn is None:
        logging.error("...")
        return
    try:
        logging.info(f"Načítání historických zápasů z {csv_path}...")
        df = pd.read_csv(csv_path)
        logging.info(f"Načteno {len(df)} záznamů.")
        df = sanitize_column_names(df)
        df.rename(
            columns={
                "teama": "team_a",
                "teamb": "team_b",
                "scorea": "score_a",
                "scoreb": "score_b",
                "ratingchangea": "rating_change_a",
                "ratingchangeb": "rating_change_b",
                "newratinga": "new_rating_a",
                "newratingb": "new_rating_b",
            },
            inplace=True,
        )
        db_columns = [
            "date",
            "team_a",
            "team_b",
            "score_a",
            "score_b",
            "tournament",
            "location",
            "rating_change_a",
            "rating_change_b",
            "new_rating_a",
            "new_rating_b",
        ]
        df_to_insert = df[[col for col in db_columns if col in df.columns]]
        logging.info("Vkládání dat historických zápasů...")
        df_to_insert.to_sql(
            "historical_matches", conn, if_exists="replace", index=False
        )
        logging.info(
            f"Tabulka 'historical_matches' úspěšně naplněna {len(df_to_insert)} záznamy."
        )
    except Exception as e:
        logging.error(f"Chyba při plnění 'historical_matches': {e}")
        logging.exception("Detail:")


if __name__ == "__main__":
    logging.info("Zahajuji inicializaci/aktualizaci struktury databáze...")
    conn = create_connection(DATABASE_PATH)
    if conn is not None:
        try:
            create_tables(conn)
            players_df = populate_players(conn, PLAYERS_CSV_PATH)
            populate_teams(conn, players_df)
            populate_historical_matches(conn, MATCHES_CSV_PATH)
            logging.info("Struktura databáze připravena/aktualizována.")
        except Exception as e:
            logging.error(f"Došlo k chybě během procesu inicializace struktury DB: {e}")
        finally:
            conn.close()
            logging.info("Připojení k databázi uzavřeno.")
    else:
        logging.error("Chyba! Nelze vytvořit připojení k databázi.")
