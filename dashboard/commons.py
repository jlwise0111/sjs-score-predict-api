import requests
from datetime import timedelta, datetime
import pandas as pd

start_year = 2010
end_year = 2023
SJS = 'SJS'

def fetch_data():
    game_data = []
    game_id = 1

    for year in range(start_year, end_year + 1):
        season = str(year) + str(year + 1)
        url = f'https://api-web.nhle.com/v1/club-schedule-season/SJS/{season}'
        response = requests.get(url)

        current_streak = 0
        prev_game_date = None

        for game in response.json()['games']:
            game_date_str = game['gameDate']
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d')

            is_home = game['homeTeam']['abbrev'] == SJS
            if is_home:
                sjs_score = game['homeTeam']['score']
                opponent_score = game['awayTeam']['score']
                opponent = game['awayTeam']['abbrev']
            else:
                sjs_score = game['awayTeam']['score']
                opponent_score = game['homeTeam']['score']
                opponent = game['homeTeam']['abbrev']

            is_win = sjs_score > opponent_score

            is_b2b = prev_game_date is not None and prev_game_date == game_date - timedelta(days=1)

            if is_win:
                current_streak += 1
            else:
                current_streak = 0

            prev_game_date = game_date
            game_data.append(
                [
                    game_id,
                    opponent,
                    is_home,
                    sjs_score,
                    opponent_score,
                    current_streak,
                    is_b2b
                ]
            )
            game_id = game_id + 1

    return pd.DataFrame(game_data, columns=[
        'id',
        'opponent',
        'is_home',
        'sjs_score',
        'opponent_score',
        'current_streak',
        'is_b2b'
    ])