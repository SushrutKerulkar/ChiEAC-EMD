"""
Data analysis and visualization module for Explain My Data.
Generates interactive Plotly charts that work across all dataset types.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

MAX_CATEGORIES = 15

PALETTE = px.colors.qualitative.Safe

# Shared layout defaults
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="#1e2130", font_size=13, font_color="white"),
)


def load_dataset(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def get_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() <= MAX_CATEGORIES
    ]
    datetime = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            if col not in datetime:
                datetime.append(col)
        except Exception:
            pass
    return numeric, categorical, datetime


# ── 1. Dataset Overview ────────────────────────────────────────────────────────
def plot_overview(df: pd.DataFrame) -> go.Figure:
    numeric, categorical, datetime_cols = get_column_types(df)
    total = len(df.columns)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Missing Values (%)", "Column Types", "Quick Stats"),
        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "table"}]],
    )

    # Missing values bar
    missing = (df.isnull().mean() * 100).sort_values(ascending=True)
    missing_nonzero = missing[missing > 0]
    if len(missing_nonzero):
        fig.add_trace(
            go.Bar(
                x=missing_nonzero.values,
                y=missing_nonzero.index,
                orientation="h",
                marker_color="#e07b54",
                hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>",
                name="Missing %",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(range=[0, 100], row=1, col=1, title_text="%")
    else:
        fig.add_annotation(
            text="No missing values!", x=0.5, y=0.5,
            xref="x domain", yref="y domain",
            showarrow=False, font=dict(size=14, color="#2ca02c"),
            row=1, col=1,
        )

    # Column type pie
    other = total - len(numeric) - len(categorical) - len(datetime_cols)
    labels, values, colors = [], [], []
    for label, val, color in [
        ("Numeric", len(numeric), "#4c72b0"),
        ("Categorical", len(categorical), "#dd8452"),
        ("Datetime", len(datetime_cols), "#55a868"),
        ("Other", other, "#c44e52"),
    ]:
        if val > 0:
            labels.append(label); values.append(val); colors.append(color)

    fig.add_trace(
        go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            hovertemplate="%{label}: %{value} columns (%{percent})<extra></extra>",
            textinfo="label+percent",
        ),
        row=1, col=2,
    )

    # Quick stats table
    stats_labels = [
        "Rows", "Columns", "Numeric cols", "Categorical cols",
        "Datetime cols", "Duplicate rows", "Missing cells",
    ]
    stats_values = [
        f"{len(df):,}", f"{total:,}", str(len(numeric)), str(len(categorical)),
        str(len(datetime_cols)), f"{df.duplicated().sum():,}",
        f"{df.isnull().sum().sum():,}",
    ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="#4c72b0",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[stats_labels, stats_values],
                fill_color=[["#1e2130", "#1a2035"] * 4],
                font=dict(color="#e8eaf6", size=12),
                align=["left", "right"],
            ),
        ),
        row=1, col=3,
    )

    fig.update_layout(title_text="Dataset Overview", height=420, **BASE_LAYOUT)
    return fig


# ── 2. Numeric Distributions ──────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame):
    numeric, _, _ = get_column_types(df)
    if not numeric:
        return None
    n = min(len(numeric), 12)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=numeric[:n],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, col in enumerate(numeric[:n]):
        r, c = divmod(i, cols_per_row)
        data = df[col].dropna()
        mean_val, median_val = data.mean(), data.median()

        fig.add_trace(
            go.Histogram(
                x=data, name=col,
                marker_color="#4c72b0", opacity=0.8,
                hovertemplate=f"<b>{col}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                showlegend=False,
                nbinsx=30,
            ),
            row=r + 1, col=c + 1,
        )
        # Mean line
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color="#e07b54", line_width=1.5,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_font_size=9,
            annotation_position="top right",
            row=r + 1, col=c + 1,
        )
        # Median line
        fig.add_vline(
            x=median_val, line_dash="dot", line_color="#2ca02c", line_width=1.5,
            annotation_text=f"Median: {median_val:.2f}",
            annotation_font_size=9,
            annotation_position="top left",
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        title_text="Numeric Distributions",
        height=320 * rows,
        **BASE_LAYOUT,
    )
    return fig


# ── 3. Box Plots ──────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame):
    numeric, _, _ = get_column_types(df)
    if not numeric:
        return None
    n = min(len(numeric), 12)

    fig = go.Figure()
    for i, col in enumerate(numeric[:n]):
        data = df[col].dropna()
        fig.add_trace(
            go.Box(
                y=data, name=col,
                marker_color=PALETTE[i % len(PALETTE)],
                boxmean="sd",
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "Q1: %{q1:.2f}<br>Median: %{median:.2f}<br>"
                    "Q3: %{q3:.2f}<br>Min: %{lowerfence:.2f}<br>Max: %{upperfence:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title_text="Box Plots — Outlier Detection",
        yaxis_title="Value",
        showlegend=False,
        height=500,
        **BASE_LAYOUT,
    )
    return fig


# ── 4. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    numeric, _, _ = get_column_types(df)
    if len(numeric) < 2:
        return None
    corr = df[numeric].corr().round(3)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig = go.Figure(
        go.Heatmap(
            z=corr_masked.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
            text=corr_masked.values.round(2),
            texttemplate="%{text}",
            hovertemplate="<b>%{y} × %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
            colorbar=dict(title="r", thickness=15),
        )
    )
    fig.update_layout(
        title_text="Correlation Heatmap",
        height=max(450, len(numeric) * 40),
        xaxis=dict(tickangle=-30),
        **BASE_LAYOUT,
    )
    return fig


# ── 5. Categorical Counts ─────────────────────────────────────────────────────
def plot_categoricals(df: pd.DataFrame):
    _, categorical, _ = get_column_types(df)
    if not categorical:
        return None
    n = min(len(categorical), 9)
    cols_per_row = 2
    rows = (n + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=categorical[:n],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    for i, col in enumerate(categorical[:n]):
        r, c = divmod(i, cols_per_row)
        vc = df[col].value_counts().head(MAX_CATEGORIES)
        pct = (vc / len(df) * 100).round(1)

        fig.add_trace(
            go.Bar(
                x=vc.values,
                y=vc.index.astype(str),
                orientation="h",
                marker_color=PALETTE[i % len(PALETTE)],
                customdata=pct.values,
                hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>Share: %{customdata:.1f}%<extra></extra>",
                showlegend=False,
                name=col,
            ),
            row=r + 1, col=c + 1,
        )
        fig.update_yaxes(autorange="reversed", row=r + 1, col=c + 1)

    fig.update_layout(
        title_text="Categorical Value Counts",
        height=350 * rows,
        **BASE_LAYOUT,
    )
    return fig


# ── 6. Time Series ────────────────────────────────────────────────────────────
def plot_timeseries(df: pd.DataFrame):
    numeric, _, datetime_cols = get_column_types(df)
    if not datetime_cols or not numeric:
        return None
    date_col = datetime_cols[0]
    try:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], infer_datetime_format=True)
        temp = temp.sort_values(date_col)
    except Exception:
        return None

    n = min(len(numeric), 4)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=numeric[:n],
        vertical_spacing=0.08,
    )

    for i, col in enumerate(numeric[:n]):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=temp[date_col], y=temp[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.8),
                fill="tozeroy",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.12)") if color.startswith("rgb") else color,
                hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>",
            ),
            row=i + 1, col=1,
        )

    fig.update_layout(
        title_text=f"Time Series (by {date_col})",
        height=280 * n,
        showlegend=False,
        **BASE_LAYOUT,
    )
    fig.update_xaxes(title_text=date_col, row=n, col=1)
    return fig


# ── 7. Scatter Matrix ─────────────────────────────────────────────────────────
def plot_pairplot(df: pd.DataFrame):
    numeric, categorical, _ = get_column_types(df)
    if len(numeric) < 2:
        return None
    cols = numeric[:5]
    color_col = categorical[0] if categorical else None

    plot_df = df[cols + ([color_col] if color_col else [])].dropna()

    fig = px.scatter_matrix(
        plot_df,
        dimensions=cols,
        color=color_col,
        color_discrete_sequence=PALETTE,
        opacity=0.6,
        title="Scatter Matrix — Numeric Relationships",
        labels={c: c for c in cols},
    )
    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=4),
        hovertemplate="<b>%{xaxis.title.text}</b>: %{x:.2f}<br><b>%{yaxis.title.text}</b>: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(height=600, **BASE_LAYOUT)
    return fig


# ── 8. Missing Value Pattern ──────────────────────────────────────────────────
def plot_missing_pattern(df: pd.DataFrame):
    if df.isnull().sum().sum() == 0:
        return None
    missing = df.isnull().astype(int)
    missing_cols = missing.columns[missing.any()].tolist()
    if not missing_cols:
        return None
    missing = missing[missing_cols]

    # Sample rows for readability
    sample = missing.sample(min(300, len(missing)), random_state=42).reset_index(drop=True)

    fig = go.Figure(
        go.Heatmap(
            z=sample.values.T,
            x=sample.index,
            y=sample.columns,
            colorscale=[[0, "#2ca02c"], [1, "#e07b54"]],
            showscale=True,
            colorbar=dict(
                tickvals=[0, 1], ticktext=["Present", "Missing"],
                thickness=15,
            ),
            hovertemplate="Row: %{x}<br>Column: %{y}<br>%{z}<extra></extra>",
            zmin=0, zmax=1,
        )
    )
    fig.update_layout(
        title_text="Missing Value Pattern (orange = missing)",
        height=max(300, len(missing_cols) * 30 + 100),
        xaxis_title="Row index (sampled)",
        **BASE_LAYOUT,
    )
    return fig


# ── Master function ───────────────────────────────────────────────────────────
def generate_all_charts(df: pd.DataFrame) -> dict:
    charts = {}
    charts["overview"] = ("Dataset Overview", plot_overview(df))

    r = plot_distributions(df)
    if r: charts["distributions"] = ("Numeric Distributions", r)

    r = plot_boxplots(df)
    if r: charts["boxplots"] = ("Box Plots — Outlier Detection", r)

    r = plot_correlation(df)
    if r: charts["correlation"] = ("Correlation Heatmap", r)

    r = plot_categoricals(df)
    if r: charts["categoricals"] = ("Categorical Value Counts", r)

    r = plot_timeseries(df)
    if r: charts["timeseries"] = ("Time Series", r)

    r = plot_pairplot(df)
    if r: charts["pairplot"] = ("Scatter Matrix", r)

    r = plot_missing_pattern(df)
    if r: charts["missing_pattern"] = ("Missing Value Pattern", r)

    return charts


def build_data_summary(df: pd.DataFrame) -> str:
    numeric, categorical, datetime_cols = get_column_types(df)
    lines = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Numeric columns ({len(numeric)}): {', '.join(numeric[:15])}",
        f"Categorical columns ({len(categorical)}): {', '.join(categorical[:15])}",
        f"Datetime columns: {', '.join(datetime_cols[:5]) or 'None'}",
        f"Duplicate rows: {df.duplicated().sum()}",
        f"Total missing cells: {df.isnull().sum().sum()} "
        f"({df.isnull().mean().mean()*100:.1f}% of all cells)",
        "",
        "--- Numeric Summary ---",
        df[numeric].describe().to_string() if numeric else "N/A",
        "",
        "--- Top Categorical Values ---",
    ]
    for col in categorical[:8]:
        top = df[col].value_counts().head(5).to_dict()
        lines.append(f"  {col}: {top}")
    return "\n".join(lines)
