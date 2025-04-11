import streamlit as st
import plotly.figure_factory as ff

def wins_losses_chart(data, option):
    st.header("Wins/Losses by location")
    st.bar_chart(
        data[data['opponent'] == option] \
            .value_counts(['is_win', 'is_home']) \
            # .rename_axis('unique_values')\
            .reset_index(name='counts'),
        x='is_win',
        y='counts',
        color='is_home'
    )

def _histogram(data, column_name):
    hist_data = [
        data \
            .loc[data['is_home'] == True, column_name].tolist()
        , data \
            .loc[data['is_home'] == False, column_name].tolist()
    ]

    group_labels = ['Home Game', 'Away Games']

    fig = ff.create_distplot(
        hist_data, group_labels)

    st.plotly_chart(fig)

def histograms(data):
    st.header("SJS Goal Histogram")
    _histogram(data, 'sjs_score')

    st.header("Opponent Goal Histogram")
    _histogram(data, 'opponent_score')

def top_5_charts(data, option):
    st.header("Top 5 games by SJS score")
    st.table(
        data \
            .loc[data['opponent'] == option] \
            .sort_values('sjs_score', ascending=False).head(5)
    )

    st.header("Top 5 games by opponent score")
    st.table(
        data \
            .loc[data['opponent'] == option] \
            .sort_values('opponent_score', ascending=False).head(5)
    )