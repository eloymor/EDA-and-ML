from __future__ import annotations

import pathlib
from typing import List

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

from sklearn.preprocessing import MinMaxScaler

# Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Spotify" / "data" / "spotify-2023.csv"
COLS_TO_CHECK = ["streams", "in_deezer_playlists", "in_shazam_charts"]


def load_spotify() -> pd.DataFrame:
    # The EDA notebook used ISO-8859-2 encoding
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-2")


    # Build release_date column if parts exist
    for part in ["released_year", "released_month", "released_day"]:
        if part not in df.columns:
            return df

    release_date = (
        df["released_year"].astype(str)
        + "/" + df["released_month"].astype(str)
        + "/" + df["released_day"].astype(str)
    )
    df["release_date"] = pd.to_datetime(release_date, yearfirst=True, errors="coerce")
    df = df.drop(columns=["released_year", "released_month", "released_day"])

    df = clean_data(df, COLS_TO_CHECK)

    return df

def clean_data(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    idx_dict = {}
    for col in cols:
        indexes = []
        for idx, row in df.iterrows():
            try:
                pd.to_numeric(row[col])
                continue
            except ValueError:
                indexes.append(idx)
        idx_dict[col] = indexes

    error_indexes = set()
    for col, indexes in idx_dict.items():
        for idx in indexes:
            s = df[col].iloc[idx]
            if len(s) < 10:  # if the string is shorter than 10 characters, it's probably a number'
                s = s.replace(",", ".")
                try:
                    n = pd.to_numeric(s)
                    # pd.to_numeric returns a numpy.float64 class
                    if isinstance(n, np.float64):
                        n = n.item()  # access the value inside the numpy.float64 class
                    df.loc[idx, col] = n
                except ValueError as e:
                    error_indexes.add(idx)
            else:
                error_indexes.add(idx)

    if error_indexes:
        df = df.drop(index=error_indexes)

    for col in COLS_TO_CHECK:
        if col.lower() == "streams":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)

    stream_scaler = MinMaxScaler()
    df["streams"] = stream_scaler.fit_transform(df["streams"].values.reshape(-1, 1))

    return df



def numeric_columns(df: pd.DataFrame) -> List[str]:
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    # Exclude identifiers that are not useful for distribution/scatter by default
    exclude = {"artist_count"}
    return [c for c in nums if c not in exclude]

df_global = load_spotify()
NUMERIC_COLS = numeric_columns(df_global)

# Some commonly analyzed features in the EDA
DEFAULT_X_OPTIONS = [
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
DEFAULT_X_OPTIONS = [c for c in DEFAULT_X_OPTIONS if c in df_global.columns]

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Spotify 2023 Dashboard"

app.layout = html.Div([
    html.H1("Spotify 2023 – Interactive Dashboard"),
    html.P("Explore the Top Spotify Songs 2023 dataset interactively."),

    html.Div([
        html.Div([
            html.Label("Year range"),
            dcc.RangeSlider(
                id="year-range",
                min=int(df_global["released_year"].min()) if "released_year" in df_global else 1900,
                max=int(df_global["released_year"].max()) if "released_year" in df_global else 2025,
                step=1,
                allowCross=False,
                value=[
                    int(df_global["released_year"].min()) if "released_year" in df_global else 1900,
                    int(df_global["released_year"].max()) if "released_year" in df_global else 2025,
                ],
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
            html.Br(),
            html.Label("Key filter"),
            dcc.Dropdown(
                id="key-filter",
                options=[
                    {"label": "All", "value": "__all__"},
                ] + (
                    [
                        {"label": f"{str(v)}", "value": v}
                        for v in sorted(df_global["key"].dropna().unique().tolist())
                    ] if "key" in df_global else []
                ),
                value="__all__",
                clearable=False,
            ),
            html.Br(),
            html.Label("Scatter: X vs streams (color by key)"),
            dcc.Dropdown(
                id="x-scatter",
                options=[{"label": c, "value": c} for c in (DEFAULT_X_OPTIONS or NUMERIC_COLS)],
                value=(DEFAULT_X_OPTIONS[0] if DEFAULT_X_OPTIONS else (NUMERIC_COLS[0] if NUMERIC_COLS else None)),
                clearable=False,
            ),
            html.Br(),
            html.Label("Histogram feature"),
            dcc.Dropdown(
                id="hist-feature",
                options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                value=(DEFAULT_X_OPTIONS[0] if DEFAULT_X_OPTIONS else (NUMERIC_COLS[0] if NUMERIC_COLS else None)),
                clearable=False,
            ),
            html.Br(),
            html.Label("Correlation method"),
            dcc.RadioItems(
                id="corr-method",
                options=[
                    {"label": "Pearson", "value": "pearson"},
                    {"label": "Spearman", "value": "spearman"},
                ],
                value="spearman",
                inline=True,
            ),
        ], style={"flex": "1", "minWidth": "260px", "paddingRight": "12px"}),

        html.Div([
            dcc.Tabs([
                dcc.Tab(label="Scatter vs Streams", children=[
                    dcc.Graph(id="scatter-streams")
                ]),
                dcc.Tab(label="Histogram", children=[
                    dcc.Graph(id="histogram")
                ]),
                dcc.Tab(label="Correlation Heatmap", children=[
                    dcc.Graph(id="corr-heatmap")
                ]),
                dcc.Tab(label="Streams Over Time", children=[
                    dcc.Graph(id="streams-over-time")
                ]),
                dcc.Tab(label="Sample Table", children=[
                    dash_table.DataTable(
                        id="sample-table",
                        page_size=10,
                        style_table={"overflowX": "auto"},
                    )
                ]),
            ])
        ], style={"flex": "3"})
    ], style={"display": "flex", "gap": "12px"}),

    html.Hr(),
    html.Div([
        html.Small(f"Data: {DATA_PATH}")
    ])
])


def filter_df(df: pd.DataFrame, year_range: List[int], key_value):
    f = df.copy()
    if "released_year" in f and year_range and len(year_range) == 2:
        f = f[(f["released_year"] >= year_range[0]) & (f["released_year"] <= year_range[1])]
    if "key" in f and key_value is not None and key_value != "__all__":
        f = f[f["key"] == key_value]
    return f


@callback(
    Output("scatter-streams", "figure"),
    Output("histogram", "figure"),
    Output("corr-heatmap", "figure"),
    Output("streams-over-time", "figure"),
    Output("sample-table", "data"),
    Output("sample-table", "columns"),
    Input("year-range", "value"),
    Input("key-filter", "value"),
    Input("x-scatter", "value"),
    Input("hist-feature", "value"),
    Input("corr-method", "value"),
)
def update_graphs(year_range, key_value, x_scatter, hist_feature, corr_method):  # noqa: D401
    dff = filter_df(df_global, year_range, key_value)

    # Scatter vs streams
    if x_scatter in dff.columns and "streams" in dff.columns:
        fig_scatter = px.scatter(
            dff,
            x=x_scatter,
            y="streams",
            color=("key" if "key" in dff.columns else None),
            labels={x_scatter: x_scatter, "streams": "Streams"},
            title=f"{x_scatter} vs Streams",
        )
    else:
        fig_scatter = go.Figure()
        fig_scatter.update_layout(title="Scatter unavailable: missing columns")

    # Histogram
    if hist_feature in dff.columns:
        fig_hist = px.histogram(
            dff,
            x=hist_feature,
            nbins=40,
            marginal="box",
            title=f"Distribution of {hist_feature}"
        )
    else:
        fig_hist = go.Figure()
        fig_hist.update_layout(title="Histogram unavailable: missing column")

    # Correlation heatmap (numeric only)
    num_df = dff.select_dtypes(include=["number"]).copy()
    if not num_df.empty:
        corr = num_df.corr(method=corr_method).round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            origin="lower",
            title=f"Correlation Heatmap ({corr_method.title()})",
            zmin=-1,
            zmax=1,
        )
    else:
        fig_corr = go.Figure()
        fig_corr.update_layout(title="Correlation unavailable: no numeric data after filters")

    # Streams over time (requires release_date)
    if "release_date" in dff.columns and "streams" in dff.columns:
        s = (
            dff.dropna(subset=["release_date"]).groupby("release_date")["streams"].sum().sort_index()
        )
        fig_time = px.line(
            s.reset_index(), x="release_date", y="streams", markers=True,
            title="Total Streams by Release Date"
        )
    else:
        fig_time = go.Figure()
        fig_time.update_layout(title="Time series unavailable: missing release_date or streams")

    # Sample table
    sample = dff.head(100)
    cols = [{"name": c, "id": c} for c in sample.columns]
    data = sample.to_dict("records")

    return fig_scatter, fig_hist, fig_corr, fig_time, data, cols


if __name__ == "__main__":
    # Expose on localhost by default
    app.run(debug=True, host="127.0.0.1", port=8050)
