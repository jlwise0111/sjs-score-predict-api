import streamlit as st
from commons import fetch_data
from charts import wins_losses_chart, histograms, top_5_charts


@st.cache_data
def get_data():
    return fetch_data()

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

[col1, col2] = st.columns(2)
# Select home vs away
home_filter = col1.radio("Choose Game Location", ["All", "Home", "Away"])
if home_filter == "Home":
    data = data[data['is_home'] == True]
elif home_filter == "Away":
    data = data[data['is_home'] == False]

# Select back-to-back or not
b2b_filter = col2.radio("Back-to-Back Game?", ["All", "Yes", "No"])
if b2b_filter == "Yes":
    data = data[data['is_b2b'] == True]
elif b2b_filter == "No":
    data = data[data['is_b2b'] == False]

wins_losses_chart(data, option)

# Histogram
if home_filter == 'All':
    histograms(data)
else:
    st.write("Please set the Game Location filter to All to see the goal histograms.")

top_5_charts(data, option)
