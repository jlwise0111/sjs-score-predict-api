import numpy as np
import requests
from datetime import timedelta, datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

def train(game_data):
    X = game_data.drop(['id', 'sjs_score', 'opponent_score'], axis=1)
    y = game_data[['sjs_score', 'opponent_score']]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_opponent = encoder.fit_transform(X[['opponent']])
    encoded_opponent_df = pd.DataFrame(encoded_opponent, columns=encoder.get_feature_names_out(['opponent']))
    X = pd.concat([X.drop('opponent', axis=1), encoded_opponent_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(max_features='sqrt')
    rf_model.fit(X_train, y_train)

    return rf_model, encoder

def predict(is_home, current_streak, is_b2b, opponent, encoder, model):
    opponent_encoded = encoder.transform([[opponent]])
    input_data = np.concatenate(([is_home, current_streak, is_b2b], opponent_encoded[0]))
    input_data = input_data.reshape(1, -1)

    predicted = model.predict(input_data)[0]
    print(predicted)
    return round(predicted[0]), round(predicted[1])