import streamlit as st
import pandas as pd

from pathlib import Path

from src.data_loader import (
    load_all_stations,
    add_target,
    remove_missing_targets,
)

from src.features import (
    add_lag_features,
    remove_missing_features,
)

import plotly.express as px

st.set_page_config(
    page_title="HeatRisk Deutschland",
    page_icon="🌡️",
    layout="wide",
)


st.title("HeatRisk Deutschland")

st.markdown(
    """
    ### Next-Day Heat-Day Risk

    Historical machine-learning project using official
    DWD climate data from five German cities.

    A **heat day** is defined as a day with a maximum
    temperature of at least **30°C**.
    """
)




data_dir = Path("data/raw")

df = load_all_stations(data_dir)
df = add_target(df)
df = remove_missing_targets(df)
df = add_lag_features(df)
df = remove_missing_features(df)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Observations",
    f"{len(df):,}",
)

col2.metric(
    "Cities",
    df["city"].nunique(),
)

col3.metric(
    "Next-day heat events",
    int(df["heat_day_next"].sum()),
)

col4.metric(
    "Heat-day rate",
    f"{df['heat_day_next'].mean() * 100:.1f}%",
)


st.subheader("Next-Day Heat Risk Over Time")

yearly = (
    df.assign(
        year=df["MESS_DATUM"].dt.year
    )
    .groupby("year")["heat_day_next"]
    .mean()
    .reset_index()
)

yearly["heat_day_rate"] = yearly["heat_day_next"] * 100

#Ai assisted visualization using Plotly Express
fig = px.line(
    yearly,
    x="year",
    y="heat_day_rate",
    labels={
        "year": "Year",
        "heat_day_rate": "Next-day heat-event rate (%)",
    },
    title="Historical Next-Day Heat-Event Rate",
)

fig.update_yaxes(ticksuffix="%")

st.plotly_chart(
    fig,
    use_container_width=True,
)
st.subheader("Model Performance")

results = pd.read_csv(
    "outputs/model_comparison.csv",
    index_col=0,
)

metrics = results[
    ["precision", "recall", "f1"]
].reset_index()

metrics = metrics.rename(
    columns={"index": "model"}
)

metrics_long = metrics.melt(
    id_vars="model",
    var_name="metric",
    value_name="score",
)

fig = px.bar(
    metrics_long,
    x="model",
    y="score",
    color="metric",
    barmode="group",
    labels={
        "model": "Model",
        "score": "Score",
        "metric": "Metric",
    },
    title="Model Performance on the Test Set",
)

fig.update_yaxes(range=[0, 1])

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.markdown(
    """
    ### How to read the model results

    The three main metrics shown above measure different things.

    **Precision** tells us how often a predicted heat day was actually
    a heat day. A higher precision means fewer false alarms.

    **Recall** tells us how many of the actual heat days the model
    successfully detected. A higher recall means the model misses
    fewer heat days.

    **F1 score** combines precision and recall into one score. It is
    useful when we want a balance between detecting heat days and
    avoiding false alarms.

    For this project, **Logistic Regression has the highest recall
    (95.6%)**. This means it detects almost all of the actual heat-day
    events, but it also produces more false alarms.

    The **Random Forest has higher precision (42.9%) and a higher
    F1 score (50.3%)**. In other words, when Random Forest predicts
    a heat day, its prediction is more often correct, and it provides
    a better balance between detecting heat days and avoiding false
    alarms. However, it misses more actual heat days than Logistic
    Regression.

    This shows an important trade-off: **if missing a heat day is the
    biggest concern, Logistic Regression is more suitable; if reducing
    false alarms is more important, Random Forest performs better.**
    """
)