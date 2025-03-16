from datetime import timedelta, datetime

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'San Jose Sharks Score and Win Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# Global constants
start_year = 2010
end_year = 2023
SJS = 'SJS'

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nhl.db'
db = SQLAlchemy(app)


# Define a database model
class SJSGame(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    opponent = db.Column(db.String(3), nullable=False)
    is_home = db.Column(db.Boolean, nullable=False)
    sjs_score = db.Column(db.Integer, nullable=False)
    opponent_score = db.Column(db.Integer, nullable=False)
    current_streak = db.Column(db.Integer, nullable=False)
    is_b2b = db.Column(db.Boolean, nullable=False)


# Create the database
with app.app_context():
    db.create_all()


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


# Global variables for model and encoder
model = None
encoder = None


@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the NHL stats API, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model, encoder

    game_data = fetch_data()

    db.session.query(SJSGame).delete()

    for _, row in game_data.iterrows():
        game = SJSGame(
            id=row['id'],
            opponent=row['opponent'],
            is_home=row['is_home'],
            sjs_score=row['sjs_score'],
            opponent_score=row['opponent_score'],
            current_streak=row['current_streak'],
            is_b2b=row['is_b2b']
        )
        db.session.add(game)
    db.session.commit()

    # Step 5: Preprocess and train model
    model, encoder = train(game_data)

    # Step 6: Generate summary statistics
    summary = {
        'total_games': len(game_data),
        'average_sjs_score': str(game_data['sjs_score'].mean()),
        'average_opponent_score': str(game_data['opponent_score'].mean()),
        'max_sjs_score': str(game_data['sjs_score'].max()),
        'max_opponent_score': str(game_data['opponent_score'].max()),
        'longest_streak': str(game_data['current_streak'].max())
    }

    return jsonify(summary)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict how badly Sharks are going to lose next game (don't get me wrong, I'm a fan of SJS)
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            opponent:
              type: string
              default: "SEA"
              minLength: 3
              maxLength: 3
            is_home:
              type: boolean
              default: true
            current_streak:
              type: integer
              default: 0
            is_b2b:
              type: boolean
              default: false

    responses:
      200:
        description: Predicted game score
    """
    global model, encoder  # Ensure that the encoder and model are available for prediction

    # Check if the model and encoder are initialized
    if model is None or encoder is None:
        return jsonify({
            "error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        opponent = data.get('opponent')
        is_home = data.get('is_home')
        current_streak = data.get('current_streak')
        is_b2b = data.get('is_b2b')

        if None in [opponent, is_home, current_streak, is_b2b]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        # Transform the input using the global encoder
        opponent_encoded = encoder.transform([[opponent]])
        input_data = np.concatenate(([is_home, current_streak, is_b2b], opponent_encoded[0]))
        input_data = input_data.reshape(1, -1)

        # Predict the price
        predicted = model.predict(input_data)[0]
        print(predicted)

        return jsonify({
            "predicted_sjs_score": round(predicted[0]),
            "predicted_opponent_score": round(predicted[1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
