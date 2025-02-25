import streamlit as st
import sqlite3
import sqlite_vec
import pandas as pd
from utils.simulator import (
    get_current_match_day,
    get_current_season_year,
    simulate_match_day,
    start_new_season,
)

# Set page configuration
st.set_page_config(page_title="Premier League Simulator", page_icon="⚽", layout="wide")


# Main application
def main():
    st.title("⚽ Premier League Simulator")

    # Loader
    status = st.sidebar.status("Loading...", expanded=True)

    # Connect to database
    conn = sqlite3.connect("premier_league.db")

    # Load sqlite vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  
    conn.enable_load_extension(False)    

    try:
        # Get current match day and season
        current_match_day = get_current_match_day(conn)
        current_season = get_current_season_year(conn)

        st.subheader(
            f"Season {current_season} - Match Day {min(current_match_day, 38)}/38"
        )

        # Create columns for buttons
        col1, col2 = st.columns([1, 1])

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

        # Create columns fixtures
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display upcoming fixtures
            st.subheader(f"📅 Upcoming Match Day {current_match_day}")

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
                st.subheader(f"🏁 Match Day {current_match_day - 1} Results")

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
        st.subheader("🏆 League Table")

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
                df_league.iloc[0, 0] = df_league.iloc[0, 0] + " 🥇"
                df_league.iloc[1, 0] = df_league.iloc[1, 0] + " 🥈"
                df_league.iloc[2, 0] = df_league.iloc[2, 0] + " 🥉"

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
