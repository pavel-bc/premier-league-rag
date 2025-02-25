import random
import json
from utils.utils import (
    get_ollama_embedding,
    random_match_prediction,
    match_embedding_text,
)

def predict_match_score(home_team, away_team, season_year, cursor):
    """Use vector similarity search to predict score based on historical data."""
    # Build a query string similar to how we'd format matches in the embeddings table
    query_text = match_embedding_text(home_team, away_team, "?", "?", season_year)

    try:
        # Check if we have any historical matches in the embeddings table
        cursor.execute("SELECT COUNT(*) FROM match_embeddings")
        count = cursor.fetchone()[0]

        if count == 0:
            # No historical matches yet, use random prediction
            return random_match_prediction()

        # Get embedding for the query using the same mxbai-embed-large model
        query_embedding = get_ollama_embedding(query_text)
        query_embedding_json = json.dumps(query_embedding.tolist())

        # Use SQLite's vec0 extension to find similar matches directly in the query
        # Filter by the same teams to get more relevant predictions
        cursor.execute(
            """
            SELECT 
                document,
                team_home, 
                team_away, 
                goals_home, 
                goals_away, 
                match_day, 
                season_year,
                round(distance, 2) as similarity_score
            FROM match_embeddings 
            WHERE team_home = ? AND team_away = ?
            AND embedding MATCH ?
            ORDER BY distance
            LIMIT 5
            """,
            (home_team, away_team, query_embedding_json),
        )

        top_matches = cursor.fetchall()

        # If no matches with the same teams, try to find matches with any teams
        if not top_matches:
            cursor.execute(
                """
                SELECT 
                    document,
                    team_home, 
                    team_away, 
                    goals_home, 
                    goals_away, 
                    match_day, 
                    season_year,
                    round(distance, 2) as similarity_score
                FROM match_embeddings 
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT 5
                """,
                (query_embedding_json,),
            )
            top_matches = cursor.fetchall()

        if not top_matches:
            # Fallback: random prediction if no similar matches found
            return random_match_prediction()

        # Calculate weights based on similarity scores (distance)
        # Lower distance means higher similarity, so invert it
        similarities = [match[7] for match in top_matches]  # index 7 is similarity_score
        total_weight = sum(similarities)

        if total_weight <= 0:
            # Avoid division by zero
            return random_match_prediction()

        weighted_home_goals = (
            sum(match[3] * match[7] for match in top_matches) / total_weight
        )
        weighted_away_goals = (
            sum(match[4] * match[7] for match in top_matches) / total_weight
        )

        # Round to nearest integer
        home_score = round(weighted_home_goals)
        away_score = round(weighted_away_goals)

        # Add some randomness based on how confident we are (average similarity)
        avg_similarity = total_weight / len(top_matches)

        # More randomness for lower similarity
        randomness_factor = avg_similarity * 2

        if random.random() < randomness_factor * 0.3:
            home_score = max(0, home_score + random.choice([-1, 1]))
        if random.random() < randomness_factor * 0.3:
            away_score = max(0, away_score + random.choice([-1, 1]))

        # Debug logging
        print(f"🔮 Query: {query_text}, based on similar matches:")
        for i, match in enumerate(top_matches[:3], 1):
            print(
                f"{i}. {match[1]} {match[3]}-{match[4]} {match[2]} {match[6]} (similarity: {match[7]:.2f})"
            )
        print()

        return home_score, away_score
    except Exception as e:
        print(f"Error predicting score with vector search: {e}")
        # Fallback to random prediction
        return random_match_prediction()


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
        # Predict score using vector similarity
        home_goals, away_goals = predict_match_score(
            home_team, away_team, current_season, cursor
        )

        status.write(f"🔮 {home_team} vs {away_team}: {home_goals}-{away_goals}")

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
        match_text = match_embedding_text(
            home_team, away_team, home_goals, away_goals, current_season
        )
        embedding = get_ollama_embedding(match_text)

        # Create document text for the match (will be the primary key)
        document_text = (
            f"{home_team} vs {away_team} ({current_season}) - {home_goals}:{away_goals}"
        )

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
                current_season,
                json.dumps(embedding.tolist()),
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
    generate_season_fixtures(cursor, teams, next_season)

    conn.commit()


def generate_season_fixtures(cursor, teams, season_year):
    """Generate a full season of fixtures with each team playing each other twice."""
    # Clear existing fixtures for the season
    cursor.execute("DELETE FROM fixtures WHERE season_year = ?", (season_year,))

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
    print("Generating 1st half-season fixtures...")
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
    print("Generating 2nd half-season fixtures...")
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
    print(f"Saving fixtures for season {season_year}...")
    for match_day, round_fixtures in enumerate(all_fixtures, 1):
        
        for home, away in round_fixtures:
            cursor.execute(
                """
            INSERT INTO fixtures (team_home, team_away, match_day, season_year, played)
            VALUES (?, ?, ?, ?, 0)
            """,
                (home, away, match_day, season_year),
            )

    print("Done!")
