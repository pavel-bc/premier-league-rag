import streamlit as st
import sqlite3
import pandas as pd
import random
import requests
import numpy as np
import os

# Model used for embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDINGS_MODEL = "mxbai-embed-large"

# Set page configuration
st.set_page_config(page_title="Premier League Simulator", page_icon="⚽", layout="wide")

# Database initialization functions
def initialize_database(status):
    """Check if database exists and create it if needed."""
    db_exists = os.path.exists("premier_league.db")

    if not db_exists:
        create_database(status)
        return True

    # If it exists, check if it has expected tables
    conn = sqlite3.connect("premier_league.db")
    cursor = conn.cursor()

    # Check for settings table
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
    )
    settings_exists = cursor.fetchone() is not None

    if not settings_exists:
        # Database exists but tables may be missing or corrupted
        conn.close()
        create_database(status)
        return True

    conn.close()
    return False


def create_database(status):
    """Create and initialize the database with tables and initial data."""
    conn = sqlite3.connect("premier_league.db")
    cursor = conn.cursor()

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

    # Create embeddings table for RAG
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS match_embeddings (
        id INTEGER PRIMARY KEY,
        team_home TEXT NOT NULL,
        team_away TEXT NOT NULL,
        goals_home INTEGER NOT NULL,
        goals_away INTEGER NOT NULL,
        match_day INTEGER NOT NULL,
        season_year INTEGER NOT NULL,
        embedding BLOB NOT NULL
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
        status.write("Creating teams...")
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
        generate_season_fixtures(cursor, teams, current_season, status)

    conn.commit()
    conn.close()


def generate_season_fixtures(cursor, teams, season_year, status):
    """Generate a full season of fixtures with each team playing each other twice."""
    # Clear existing fixtures for the season
    cursor.execute("DELETE FROM fixtures WHERE season_year = ?", (season_year,))

    # Number of teams
    num_teams = len(teams)

    # To ensure each team plays each other twice (home and away)
    num_fixtures_per_team = 2 * (
        num_teams - 1
    )  # Each team plays every other team twice
    total_fixtures = num_teams * num_fixtures_per_team // 2  # Total number of fixtures

    # We need to schedule these fixtures across multiple match days
    # Each team can play only once per match day, and there are 20 teams,
    # so we need 10 fixtures per match day
    fixtures_per_match_day = num_teams // 2
    total_match_days = (
        total_fixtures // fixtures_per_match_day
    )  # Should be 38 for 20 teams

    # Generate fixtures using the "round-robin" algorithm
    # This algorithm guarantees each team plays once per match day
    # and plays against each other team twice (home and away)

    # Create a copy of teams list to manipulate
    teams_circle = teams.copy()

    # If odd number of teams, add a "bye" team
    if len(teams_circle) % 2 != 0:
        teams_circle.append("BYE")

    # Number of teams after handling odd case
    n = len(teams_circle)

    # Generate fixtures for the first half of the season (each team plays each other once)
    status.write("Generating 1st half-season fixtures...")
    fixtures_first_half = []
    for round_num in range(n - 1):
        round_fixtures = []
        for i in range(n // 2):
            team1 = teams_circle[i]
            team2 = teams_circle[n - 1 - i]

            # Skip fixtures involving the "BYE" team
            if team1 != "BYE" and team2 != "BYE":
                # Alternate home/away to ensure balanced schedule
                if round_num % 2 == 0:
                    round_fixtures.append((team1, team2))
                else:
                    round_fixtures.append((team2, team1))

        fixtures_first_half.append(round_fixtures)

        # Rotate teams (keep first team fixed, rotate the rest)
        teams_circle = [teams_circle[0]] + [teams_circle[-1]] + teams_circle[1:-1]

    # Second half of the season - reverse the home/away for each fixture
    status.write("Generating 2nd half-season fixtures...")
    fixtures_second_half = []
    for round_fixtures in fixtures_first_half:
        reverse_fixtures = [(away, home) for home, away in round_fixtures]
        fixtures_second_half.append(reverse_fixtures)

    # Combine first and second half
    all_fixtures = fixtures_first_half + fixtures_second_half

    # Shuffle the order of match days for more variety while maintaining
    # the constraint that each team plays once per match day
    random.shuffle(all_fixtures)

    # Insert fixtures into database with match day numbers
    for match_day, round_fixtures in enumerate(all_fixtures, 1):
        status.write(
            f"Savings fixtures for matchday {match_day} season {season_year}..."
        )
        for home, away in round_fixtures:
            cursor.execute(
                """
            INSERT INTO fixtures (team_home, team_away, match_day, season_year, played)
            VALUES (?, ?, ?, ?, 0)
            """,
                (home, away, match_day, season_year),
            )


def get_ollama_embedding(text):
    """Get embeddings from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDINGS_MODEL, "prompt": text},
        )
        return np.array(response.json()["embedding"])
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return np.zeros(1024)  # Return zero vector as fallback


def predict_match_score(home_team, away_team, match_day, season_year, cursor, status):
    """Use Ollama to predict match score based on historical data."""
    # Get relevant historical matches as context
    cursor.execute(
        """
    SELECT team_home, team_away, goals_home, goals_away, match_day, season_year
    FROM fixtures
    WHERE played = 1 AND ((team_home = ? AND team_away = ?) OR (team_home = ? AND team_away = ?))
    ORDER BY season_year DESC, match_day DESC
    LIMIT 10
    """,
        (home_team, away_team, away_team, home_team),
    )

    historical_matches = cursor.fetchall()

    # Create context for prediction
    context = "Based on the following previous match results, predict the score for the upcoming match:\n\n"

    if historical_matches:
        for match in historical_matches:
            h_team, a_team, h_goals, a_goals, m_day, s_year = match
            context += f"{h_team} {h_goals}-{a_goals} {a_team} (Season {s_year}, Match Day {m_day})\n"
    else:
        context += "No previous match data available.\n"

    # Get current league standings for context
    cursor.execute(
        """
    SELECT team_name, matches_played, wins, draws, losses, goals_for, goals_against, points
    FROM league_table
    WHERE season_year = ? AND (team_name = ? OR team_name = ?)
    """,
        (season_year, home_team, away_team),
    )

    standings = cursor.fetchall()
    context += "\nCurrent league standings for these teams:\n"
    for team in standings:
        name, played, wins, draws, losses, gf, ga, pts = team
        context += f"{name}: Played {played}, {wins}W {draws}D {losses}L, GF {gf}, GA {ga}, Points {pts}\n"

    # Question for Ollama
    prompt = (
        context
        + f"\n\nPredict the score for {home_team} vs {away_team} in Season {season_year}, Match Day {match_day}. Give only the score in format 'X-Y' where X is home goals and Y is away goals."
    )
    status.write(f"🔮 {home_team} vs {away_team}...")
    print(f"Sending a prompt {prompt}")

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDINGS_MODEL, "prompt": prompt, "stream": False},
        )

        # Extract the prediction
        response_json = response.json()
        print(response_json)
        result = response_json["response"].strip()

        # Try to find a score pattern (like "2-1" or "1-0")
        import re

        score_pattern = re.search(r"(\d+)-(\d+)", result)
        if score_pattern:
            home_score = int(score_pattern.group(1))
            away_score = int(score_pattern.group(2))
        else:
            # If no clear pattern, generate random score as fallback
            home_score = random.randint(0, 5)
            away_score = random.randint(0, 5)

        return home_score, away_score
    except Exception as e:
        st.error(f"Ollama error: {response_json}")
        # Fallback to random prediction
        return random.randint(0, 5), random.randint(0, 5)


def get_current_match_day(conn):
    """Get the current match day."""
    cursor = conn.cursor()

    # First ensure settings table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
    )
    if cursor.fetchone() is None:
        return 1  # Default to 1 if settings table doesn't exist

    # Get current season
    cursor.execute(
        "SELECT setting_value FROM settings WHERE setting_name = 'current_season_year'"
    )
    season_result = cursor.fetchone()

    if season_result is None:
        return 1  # Default to 1 if setting doesn't exist

    current_season = int(season_result[0])

    # Check if fixtures table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fixtures'"
    )
    if cursor.fetchone() is None:
        return 1  # Default to 1 if fixtures table doesn't exist

    # Get the next unplayed match day
    cursor.execute(
        """
    SELECT MIN(match_day) FROM fixtures 
    WHERE played = 0 AND season_year = ?
    """,
        (current_season,),
    )

    result = cursor.fetchone()

    if result[0] is None:
        return 39  # All match days played

    return result[0]


def get_current_season_year(conn):
    """Get the current season year from settings."""
    cursor = conn.cursor()

    # Check if settings table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
    )
    if cursor.fetchone() is None:
        return 2025  # Default to 2025 if settings table doesn't exist

    # Get current season
    cursor.execute(
        "SELECT setting_value FROM settings WHERE setting_name = 'current_season_year'"
    )
    result = cursor.fetchone()

    if result is None:
        return 2025  # Default value

    return int(result[0])


def simulate_match_day(conn, status):
    """Simulate the next match day."""
    cursor = conn.cursor()
    current_season = get_current_season_year(conn)
    match_day = get_current_match_day(conn)

    if match_day > 38:
        st.error("All match days have been played for this season.")
        return

    # Get fixtures for the current match day
    cursor.execute(
        """
    SELECT id, team_home, team_away FROM fixtures 
    WHERE match_day = ? AND season_year = ? AND played = 0
    """,
        (match_day, current_season),
    )

    fixtures = cursor.fetchall()

    for fixture_id, home_team, away_team in fixtures:
        # Predict score using Ollama
        home_goals, away_goals = predict_match_score(
            home_team, away_team, match_day, current_season, cursor, status
        )

        # Update fixture with result
        cursor.execute(
            """
        UPDATE fixtures 
        SET goals_home = ?, goals_away = ?, played = 1 
        WHERE id = ?
        """,
            (home_goals, away_goals, fixture_id),
        )

        # Update league table
        # Home team updates
        points_home = (
            3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
        )
        cursor.execute(
            """
        UPDATE league_table 
        SET matches_played = matches_played + 1,
            wins = wins + ?,
            draws = draws + ?,
            losses = losses + ?,
            goals_for = goals_for + ?,
            goals_against = goals_against + ?,
            goal_difference = goal_difference + ?,
            points = points + ?
        WHERE team_name = ? AND season_year = ?
        """,
            (
                1 if home_goals > away_goals else 0,
                1 if home_goals == away_goals else 0,
                1 if home_goals < away_goals else 0,
                home_goals,
                away_goals,
                home_goals - away_goals,
                points_home,
                home_team,
                current_season,
            ),
        )

        # Away team updates
        points_away = (
            3 if away_goals > home_goals else (1 if away_goals == home_goals else 0)
        )
        cursor.execute(
            """
        UPDATE league_table 
        SET matches_played = matches_played + 1,
            wins = wins + ?,
            draws = draws + ?,
            losses = losses + ?,
            goals_for = goals_for + ?,
            goals_against = goals_against + ?,
            goal_difference = goal_difference + ?,
            points = points + ?
        WHERE team_name = ? AND season_year = ?
        """,
            (
                1 if away_goals > home_goals else 0,
                1 if away_goals == home_goals else 0,
                1 if away_goals < home_goals else 0,
                away_goals,
                home_goals,
                away_goals - home_goals,
                points_away,
                away_team,
                current_season,
            ),
        )

        # Store match result embedding for RAG
        match_text = f"{home_team} {home_goals}-{away_goals} {away_team} (Season {current_season}, Match Day {match_day})"
        embedding = get_ollama_embedding(match_text)

        cursor.execute(
            """
        INSERT INTO match_embeddings 
        (team_home, team_away, goals_home, goals_away, match_day, season_year, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                home_team,
                away_team,
                home_goals,
                away_goals,
                match_day,
                current_season,
                embedding.tobytes(),
            ),
        )

    conn.commit()


def start_new_season(conn, status):
    """Start a new season with fresh fixtures."""
    cursor = conn.cursor()
    current_season = get_current_season_year(conn)
    next_season = current_season + 1

    # Update the current season setting
    cursor.execute(
        "UPDATE settings SET setting_value = ? WHERE setting_name = 'current_season_year'",
        (str(next_season),),
    )

    # Get teams from the current season
    cursor.execute(
        "SELECT team_name FROM league_table WHERE season_year = ?", (current_season,)
    )
    teams = [row[0] for row in cursor.fetchall()]

    # Clear league table for the new season
    for team in teams:
        cursor.execute(
            """
        INSERT INTO league_table 
        (team_name, matches_played, wins, draws, losses, goals_for, goals_against, goal_difference, points, season_year)
        VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0, ?)
        """,
            (team, next_season),
        )

    # Generate fixtures for the new season
    generate_season_fixtures(cursor, teams, next_season, status)

    conn.commit()


def reset_database(status):
    """Reset the database to initial state."""
    status.write("Resetting DB...")
    if os.path.exists("premier_league.db"):
        os.remove("premier_league.db")
    create_database(status)


# Main application
def main():
    st.title("⚽ Premier League Simulator")

    # Loader
    status = st.sidebar.status("Loading...", expanded=True)

    # Initialize database if needed
    db_was_created = initialize_database(status)
    if db_was_created:
        status.update(label="DB creation complete!", state="complete", expanded=False)
        st.success("✅ Database created successfully.")

    # Connect to database
    conn = sqlite3.connect("premier_league.db")

    try:
        # Get current match day and season
        current_match_day = get_current_match_day(conn)
        current_season = get_current_season_year(conn)

        st.subheader(
            f"Season {current_season} - Match Day {min(current_match_day, 38)}/38"
        )

        # Create columns for buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Simulate next day button
            if current_match_day <= 38:
                if st.button(
                    f"⏩ Simulate Day {current_match_day}",
                    type="primary",
                    use_container_width=True,
                ):
                    simulate_match_day(conn, status)
                    st.rerun()

        with col2:
            # Start new season button
            if current_match_day > 38:
                if st.button(
                    "🆕 Start New Season", type="primary", use_container_width=True
                ):
                    start_new_season(conn, status)
                    st.rerun()

        with col3:
            # Reset button (red)
            if st.button(
                "🗑️ Reset Database", type="secondary", use_container_width=True
            ):
                reset_database(status)
                status.update(
                    label="DB reset complete!", state="complete", expanded=False
                )
                st.rerun()

        # Create columns fixtures
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display upcoming fixtures
            st.subheader(f"Upcoming Match Day {current_match_day}")

            if current_match_day <= 38:
                cursor = conn.cursor()
                cursor.execute(
                    """
              SELECT team_home, team_away FROM fixtures 
              WHERE match_day = ? AND season_year = ? AND played = 0
              """,
                    (current_match_day, current_season),
                )

                upcoming_fixtures = cursor.fetchall()

                if upcoming_fixtures:
                    df_upcoming = pd.DataFrame(
                        upcoming_fixtures, columns=["Home Team", "Away Team"]
                    )
                    df_upcoming.index = range(1, len(df_upcoming) + 1)
                    st.table(df_upcoming)
                else:
                    st.info("No upcoming fixtures for this match day.")
            else:
                st.info("Season completed. Click 'Start New Season' to continue.")

        # Display previous match day results
        with col2:
            if current_match_day > 1:
                st.subheader(f"Match Day {current_match_day - 1} Results")

                cursor = conn.cursor()
                cursor.execute(
                    """
              SELECT team_home, team_away, concat(goals_home, '-', goals_away) as score FROM fixtures 
              WHERE match_day = ? AND season_year = ? AND played = 1
              """,
                    (current_match_day - 1, current_season),
                )

                previous_fixtures = cursor.fetchall()

                if previous_fixtures:
                    df_previous = pd.DataFrame(
                        previous_fixtures, columns=["Home Team", "Away Team", "Score"]
                    )
                    df_previous.index = range(1, len(df_previous) + 1)
                    st.table(df_previous)

        # Display league table
        st.subheader("League Table")

        cursor = conn.cursor()
        cursor.execute(
            """
        SELECT team_name, matches_played, wins, draws, losses, goals_for, goals_against, goal_difference, points
        FROM league_table 
        WHERE season_year = ?
        ORDER BY points DESC, wins DESC, goal_difference DESC, goals_for DESC
        """,
            (current_season,),
        )

        league_data = cursor.fetchall()

        if league_data:
            df_league = pd.DataFrame(
                league_data,
                columns=[
                    "Team",
                    "Played",
                    "Won",
                    "Drawn",
                    "Lost",
                    "GF",
                    "GA",
                    "GD",
                    "Points",
                ],
            )

            # Add trophy to champion if it's end of season
            if current_match_day > 38:
                df_league.iloc[0, 0] = df_league.iloc[0, 0] + " 🏆"

            # Highlight the first place team
            def highlight_first_place(s):
                is_first_place = pd.Series(data=False, index=s.index)
                is_first_place.iloc[0] = True
                return [
                    "background-color: #008080" if v else "" for v in is_first_place
                ]

            df_league.index = range(1, len(df_league) + 1)
            st.table(df_league.style.apply(highlight_first_place, axis=0))

        status.update(label="Done!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Try clicking the 'Reset Database' button to fix database issues.")

    finally:
        # Close connection
        conn.close()


if __name__ == "__main__":
    main()
