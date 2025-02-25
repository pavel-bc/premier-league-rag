import sqlite3
import sqlite_vec
import os
import time
import json
from utils.simulator import generate_season_fixtures
from utils.utils import (
    get_ollama_embedding,
    match_embedding_text,
)

HISTORICAL_FIXTURES_CSV = "premier-league-matches.csv"
HISTORICAL_FIXTURES_LIMIT = 99999


def create_database():
    """Create and initialize the database with tables and initial data."""
    conn = sqlite3.connect("premier_league.db")
    cursor = conn.cursor()

    # Load sqlite vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  
    conn.enable_load_extension(False)  

    # Create league_table table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS league_table (
        id INTEGER PRIMARY KEY,
        team_name TEXT NOT NULL,
        matches_played INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        goals_for INTEGER DEFAULT 0,
        goals_against INTEGER DEFAULT 0,
        goal_difference INTEGER DEFAULT 0,
        points INTEGER DEFAULT 0,
        season_year INTEGER NOT NULL
    )
    """
    )

    # Create fixtures table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS fixtures (
        id INTEGER PRIMARY KEY,
        team_home TEXT NOT NULL,
        team_away TEXT NOT NULL,
        goals_home INTEGER,
        goals_away INTEGER,
        match_day INTEGER NOT NULL,
        season_year INTEGER NOT NULL,
        played BOOLEAN DEFAULT 0
    )
    """
    )

    # Create settings table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY,
        setting_name TEXT NOT NULL,
        setting_value TEXT NOT NULL
    )
    """
    )

    # Create embeddings table for RAG using vec0 extension
    cursor.execute(
        """
    CREATE VIRTUAL TABLE IF NOT EXISTS match_embeddings USING vec0(
        document TEXT PRIMARY KEY,
        team_home TEXT,
        team_away TEXT,
        goals_home INTEGER,
        goals_away INTEGER,
        match_day INTEGER,
        season_year INTEGER,
        embedding FLOAT[1024]
    )
    """
    )

    # Check if settings already exist
    cursor.execute(
        "SELECT COUNT(*) FROM settings WHERE setting_name = 'current_season_year'"
    )
    settings_exist = cursor.fetchone()[0] > 0

    if not settings_exist:
        # Insert initial settings if they don't exist
        cursor.execute(
            "INSERT INTO settings (id, setting_name, setting_value) VALUES (1, 'current_season_year', '2025')"
        )

    # Load historical match data from CSV if it exists
    try:
        if os.path.exists(HISTORICAL_FIXTURES_CSV):
            # Read CSV file
            import csv

            historical_matches_added = 0

            with open(HISTORICAL_FIXTURES_CSV, "r", encoding="utf-8") as csv_file:
                csv_reader = csv.DictReader(csv_file)

                # Process each row
                for row in csv_reader:
                    try:
                        season_year = int(row.get("Season_End_Year", 0))
                        match_day = int(row.get("Wk", 0))
                        home_team = row.get("Home", "")
                        away_team = row.get("Away", "")
                        home_goals = int(row.get("HomeGoals", 0))
                        away_goals = int(row.get("AwayGoals", 0))

                        # Skip incomplete data
                        if not home_team or not away_team:
                            continue

                        # Insert historical match data
                        cursor.execute(
                            """
                        INSERT INTO fixtures
                        (team_home, team_away, goals_home, goals_away, match_day, season_year, played)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                        """,
                            (
                                home_team,
                                away_team,
                                home_goals,
                                away_goals,
                                match_day,
                                season_year,
                            ),
                        )
                        historical_matches_added += 1

                        if historical_matches_added % 100 == 0:
                            print(
                                f"Inserted {historical_matches_added} historical fixtures"
                            )

                        if historical_matches_added >= HISTORICAL_FIXTURES_LIMIT:
                            print(f"Reached import limit, breaking...")
                            break

                        # Store match result embedding for RAG
                        match_text = match_embedding_text(
                            home_team, away_team, home_goals, away_goals, season_year
                        )
                        embedding = get_ollama_embedding(match_text)

                        # Create document text for the match (will be the primary key)
                        document_text = f"{home_team} vs {away_team} ({season_year}) - {home_goals}:{away_goals}"

                        # Insert into the vec0 table
                        cursor.execute(
                            """
                        INSERT OR REPLACE INTO match_embeddings 
                        (document, team_home, team_away, goals_home, goals_away, match_day, season_year, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                document_text,
                                home_team,
                                away_team,
                                home_goals,
                                away_goals,
                                match_day,
                                season_year,
                                json.dumps(embedding.tolist()),
                            ),
                        )
                    except Exception as e:
                        print(f"Error processing match data: {e}")

            print(f"Imported {historical_matches_added} historical matches from CSV")
    except Exception as e:
        print(f"Error loading historical match data: {e}")

    # Insert teams for the current season
    teams = [
        "Arsenal",
        "Aston Villa",
        "Bournemouth",
        "Brentford",
        "Brighton",
        "Chelsea",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Leicester City",
        "Liverpool",
        "Manchester City",
        "Manchester United",
        "Newcastle United",
        "Nottingham Forest",
        "Southampton",
        "Tottenham Hotspur",
        "West Ham United",
        "Wolverhampton",
        "Ipswich Town",
    ]

    current_season = 2025

    # Check if league table is already populated for the current season
    cursor.execute(
        "SELECT COUNT(*) FROM league_table WHERE season_year = ?", (current_season,)
    )
    teams_exist = cursor.fetchone()[0] > 0

    if not teams_exist:
        # Insert teams only if they don't exist
        print("Creating teams...")
        for team in teams:
            cursor.execute(
                """
            INSERT INTO league_table 
            (team_name, matches_played, wins, draws, losses, goals_for, goals_against, goal_difference, points, season_year)
            VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0, ?)
            """,
                (team, current_season),
            )

    # Check if fixtures are already generated for the current season
    cursor.execute(
        "SELECT COUNT(*) FROM fixtures WHERE season_year = ?", (current_season,)
    )
    fixtures_exist = cursor.fetchone()[0] > 0

    if not fixtures_exist:
        # Generate fixtures only if they don't exist
        generate_season_fixtures(cursor, teams, current_season)

    conn.commit()
    conn.close()


def main():
    start_time = time.time()
    create_database()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()
