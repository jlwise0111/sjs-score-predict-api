import streamlit as st
from commons import fetch_data, train, predict
from charts import wins_losses_chart, histograms, top_5_charts


@st.cache_data
def get_data():
    return fetch_data()

@st.cache_data
def train_model():
    return train(get_data())

@st.cache_data
def get_prediction(is_home, current_streak, is_b2b, opponent):
    model, encoder = train_model()
    return predict(is_home, current_streak, is_b2b, opponent, encoder, model)


st.set_page_config(
    page_title="San Jose Sharks Game Insights",
    page_icon="./dashboard/favicon.ico"
)

st.title("🏒 San Jose Sharks Game Insights")

data = get_data()
data['is_win'] = data['sjs_score'] > data['opponent_score']


# Filter by opponent
option = st.selectbox(
    "What opponent would you like to select?",
    data['opponent'].unique(),
)

[is_home_option_col, is_b2b_option_col, streak_option_col] = st.columns(3)
# Select home vs away
home_filter = is_home_option_col.radio("Choose Game Location", ["All", "Home", "Away"])
if home_filter == "Home":
    data = data[data['is_home'] == True]
elif home_filter == "Away":
    data = data[data['is_home'] == False]

# Select back-to-back or not
b2b_filter = is_b2b_option_col.radio("Back-to-Back Game?", ["All", "Yes", "No"])
if b2b_filter == "Yes":
    data = data[data['is_b2b'] == True]
elif b2b_filter == "No":
    data = data[data['is_b2b'] == False]

current_streak = streak_option_col.number_input(
    "Current Streak (only for score prediction)",
    min_value=0,
    max_value=10
)


st.header("Prediction")
st.write("If is_home is selected as All, then default will be home game for the prediction.")
st.write("If is_b2b is selected as All, then default will non-back-to-back game.")
[sjs_col, opponent_col] = st.columns(2)

sjs_sc, opponent_sc = get_prediction(
    is_home=home_filter != 'Away',
    is_b2b=b2b_filter == 'Yes',
    current_streak=current_streak,
    opponent=option
)
sjs_col.metric(label="SJS Score", value=sjs_sc)
opponent_col.metric(label="Opponent Score", value=opponent_sc)


wins_losses_chart(data, option)

# Histogram
if home_filter == 'All':
    histograms(data)
else:
    st.write("Please set the Game Location filter to All to see the goal histograms.")

top_5_charts(data, option)
