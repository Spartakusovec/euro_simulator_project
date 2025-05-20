import pandas as pd
import sqlite3
import os
import ast
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATABASE_PATH = os.path.join(BASE_DIR, "data", "database.db")

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
    ],  # Opraveno - přidán CF
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


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Úspěšně připojeno k {db_file}")
    except sqlite3.Error as e:
        logging.error(f"Chyba při připojování k databázi: {e}")
    return conn


def load_player_data(conn):
    try:
        query = "SELECT player_name, nationality, primary_position, secondary_positions, backup_positions, overall_rating FROM players"
        df = pd.read_sql_query(query, conn)
        logging.info(f"Načteno {len(df)} hráčů z databáze.")

        for col in ["secondary_positions", "backup_positions"]:
            df[col] = df[col].fillna("[]")
            try:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            except (ValueError, TypeError, SyntaxError) as e:
                logging.warning(
                    f"Chyba při parsování sloupce '{col}': {e}. Některé hodnoty nemusí být seznamy."
                )
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        def add_primary_to_secondary(row):
            primary = row["primary_position"]
            secondary = row["secondary_positions"]
            if isinstance(secondary, list):
                if primary not in secondary:
                    secondary.insert(0, primary)
            else:
                secondary = [primary]
            return secondary

        df["secondary_positions"] = df.apply(add_primary_to_secondary, axis=1)

        return df
    except Exception as e:
        logging.error(f"Chyba při načítání dat hráčů z DB: {e}")
        logging.exception("Detail chyby:")
        return None


def can_play(player_positions, target_position):
    if isinstance(player_positions, list):
        return target_position in player_positions
    return False


def find_best_team(formation_positions, players_df):
    local_players_df = players_df.copy()
    best_team_players = []
    players_in_team_indices = set()
    current_total_rating = 0

    for position in formation_positions:
        best_player_index = None
        best_player_rating = -1
        rating_adjustment = 0

        eligible_secondary = local_players_df[
            local_players_df.apply(
                lambda row: can_play(row["secondary_positions"], position), axis=1
            )
            & (~local_players_df.index.isin(players_in_team_indices))
        ]
        if not eligible_secondary.empty:
            best_player_index = eligible_secondary["overall_rating"].idxmax()
            best_player_rating = eligible_secondary.loc[
                best_player_index, "overall_rating"
            ]
            rating_adjustment = 0
        else:
            eligible_backup = local_players_df[
                local_players_df.apply(
                    lambda row: can_play(row["backup_positions"], position), axis=1
                )
                & (~local_players_df.index.isin(players_in_team_indices))
            ]
            if not eligible_backup.empty:
                best_player_index = eligible_backup["overall_rating"].idxmax()
                best_player_rating = eligible_backup.loc[
                    best_player_index, "overall_rating"
                ]
                rating_adjustment = -1
            else:
                logging.warning(
                    f"Pro pozici '{position}' nebyl nalezen žádný vhodný hráč."
                )
                return None, 0

        players_in_team_indices.add(best_player_index)

        player_info = {
            "player_name": local_players_df.loc[best_player_index, "player_name"],
            "assigned_position": position,
            "overall_rating_in_position": best_player_rating + rating_adjustment,
        }
        best_team_players.append(player_info)
        current_total_rating += player_info["overall_rating_in_position"]

    if len(best_team_players) != 11:
        logging.warning(
            f"Podařilo se sestavit tým pouze s {len(best_team_players)} hráči."
        )
        return None, 0

    best_team_df = pd.DataFrame(best_team_players)
    return best_team_df, current_total_rating


def main():
    logging.info("Zahajuji tvorbu soupisek...")
    conn = create_connection(DATABASE_PATH)
    if conn is None:
        logging.error("Nepodařilo se připojit k databázi. Skript končí.")
        return

    all_players_df = load_player_data(conn)
    if all_players_df is None:
        logging.error("Nepodařilo se načíst data hráčů. Skript končí.")
        conn.close()
        return

    all_players_df.reset_index(inplace=True)
    all_players_df.rename(columns={"index": "original_index"}, inplace=True)

    try:
        nationalities = all_players_df["nationality"].unique()
        logging.info(f"Nalezeno {len(nationalities)} unikátních národností.")
    except KeyError:
        logging.error(
            "Sloupec 'nationality' nebyl nalezen v datech hráčů po načtení z DB."
        )
        conn.close()
        return

    all_best_teams_list = []

    for nationality in nationalities:
        logging.info(f"Zpracovávám národnost: {nationality}")
        nationality_data = all_players_df[
            all_players_df["nationality"] == nationality
        ].copy()
        nationality_data.set_index("original_index", inplace=True)

        best_formation_name = None
        best_team_df_for_nation = None
        max_team_rating = -1

        for formation_name, formation_positions in formations.items():
            logging.debug(f"Testuji formaci {formation_name} pro {nationality}")
            team_df, team_rating = find_best_team(
                formation_positions, nationality_data.copy()
            )

            if team_df is not None and team_rating > max_team_rating:
                max_team_rating = team_rating
                best_team_df_for_nation = team_df
                best_formation_name = formation_name
                logging.debug(
                    f"Nová nejlepší formace pro {nationality}: {formation_name} (Rating: {team_rating})"
                )

        if best_team_df_for_nation is not None:
            best_team_df_for_nation["nationality"] = nationality
            best_team_df_for_nation["formation_name"] = best_formation_name
            all_best_teams_list.append(best_team_df_for_nation)
            logging.info(
                f"Nejlepší formace pro {nationality}: {best_formation_name} (Celkový rating: {max_team_rating})"
            )
        else:
            logging.warning(
                f"Pro národnost {nationality} se nepodařilo sestavit žádný platný tým."
            )

    if all_best_teams_list:
        final_rosters_df = pd.concat(all_best_teams_list, ignore_index=True)
        logging.info(
            f"Celkem vygenerováno {len(final_rosters_df)} záznamů v soupiskách pro {len(all_best_teams_list)} týmů."
        )

        try:
            logging.info(
                "Ukládání vygenerovaných soupisek do tabulky 'rosters' (nahrazení stávajících)..."
            )
            columns_for_db = [
                "nationality",
                "player_name",
                "assigned_position",
                "overall_rating_in_position",
                "formation_name",
            ]
            final_rosters_df_to_db = final_rosters_df[columns_for_db]

            final_rosters_df_to_db.to_sql(
                "rosters", conn, if_exists="replace", index=False
            )
            logging.info("Tabulka 'rosters' úspěšně naplněna.")
        except Exception as e:
            logging.error(f"Chyba při ukládání soupisek do databáze: {e}")
            logging.exception("Detail chyby:")
    else:
        logging.warning("Nebyly vygenerovány žádné soupisky k uložení.")

    conn.close()
    logging.info("Připojení k databázi uzavřeno.")


if __name__ == "__main__":
    main()
