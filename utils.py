"""
This file contains utilities to
scrape play-by-play information from
TGL glof matches
"""

import pandas as pd

def parse_json_data(json_obj: dict):
    """
    This function parses the play-by-play
    data from TGL matches 
    
    Args
        - json_obj (dict): Dictionary of
            play-by-play data for a TGL match

    Returns:
        - pbp_df (pd.DataFrame): DataFrame of
            play-by-play data for TGL matches
    """

    session_list = json_obj["data"]["playByPlayList"]["sessions"]
    sessions_df = pd.DataFrame()
    holes_df = pd.DataFrame()
    shots_df = pd.DataFrame()
    for session in session_list:
        session_id = session["sessionId"]
        sequence = session["sequence"]
        session_score = session["sessionScore"]

        session_df = pd.DataFrame(
            {
                "session_id": [session_id],
                "sequence": [sequence],
                "session_score": [session_score]
            }
        )
        sessions_df = pd.concat([sessions_df, session_df])

        holes = session["playByPlay"]
        for hole in holes:
            hole_number = hole["holeNumber"]
            hole_score = hole["holeScore"]
            winning_team_id = hole["holeWinningTeamId"]
            losing_team_id = hole["holeLosingTeamId"]
            shot_df = pd.DataFrame(hole["timeline"]).sort_values("shot", ascending=True)
            shot_df["hole_number"] = hole_number
        
            hole_df = pd.DataFrame(
                {
                    "hole_number": [hole_number],
                    "hole_score": [hole_score],
                    "session_id": [session_id],
                    "sequence": [sequence],
                    "winning_team_id": [winning_team_id],
                    "losing_team_id": [losing_team_id]
                }
            )
            shots_df = pd.concat([shots_df, shot_df])
            holes_df = pd.concat([holes_df, hole_df])

    return sessions_df, holes_df, shots_df