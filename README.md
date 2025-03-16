
# San Jose Sharks Score Prediction API

This is a Flask-based API that predicts SJS and their opponents' scores on several factors like their opponent, whether they are the home team, if they are on a win streak, and if they had a game the day before. The API has two main endpoints:
- `/reload`: Reloads the data and trains the model.
- `/predict`: Predicts the game score for a given game.

## Data Source and Prediction Process

### Data Source

The data used for this project comes from [NHL API](api-web.nhle.com).

### Prediction Process

The application uses a simple **Random Forest Model** to predict the score based on various input features.

Using this model, the app can provide a Sharks vs opponent score based on the inputs.
