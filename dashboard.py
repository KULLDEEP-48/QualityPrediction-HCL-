"""
Streamlit dashboard for live mining-process quality predictions.

Usage:
    streamlit run dashboard.py
"""

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import LATEST_DATA_PATH, PREDICTIONS_LOG_PATH

st.set_page_config(
    page_title="P_220 | Process Intelligence",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #e8f0f6;
        --muted: #91a6b8;
        --teal: #42e0c1;
        --line: rgba(149, 184, 202, .16);
        --panel: rgba(13, 29, 43, .82);
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 82% 0%, rgba(12, 115, 111, .20), transparent 33%),
            radial-gradient(ellipse at 4% 28%, rgba(34, 91, 125, .18), transparent 35%),
            linear-gradient(145deg, #07131e 0%, #0a1b29 52%, #0b202b 100%);
    }
    [data-testid="stHeader"] { background: rgba(7, 19, 30, .72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1c29 0%, #101f2b 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.7rem 2rem;
        margin-bottom: 1.35rem;
        border: 1px solid rgba(66, 224, 193, .24);
        border-radius: 18px;
        background:
            linear-gradient(105deg, rgba(12, 33, 46, .96), rgba(12, 43, 53, .77)),
            repeating-linear-gradient(135deg, transparent 0 18px, rgba(255,255,255,.02) 18px 19px);
        box-shadow: 0 18px 50px rgba(0, 0, 0, .22);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: 4%;
        top: -150px;
        border: 1px solid rgba(66, 224, 193, .22);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(66,224,193,.025), 0 0 0 58px rgba(66,224,193,.025);
    }
    .eyebrow {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .19em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .35rem 0 .3rem;
        color: #f2f8fb;
        font-size: clamp(1.6rem, 3vw, 2.35rem);
        letter-spacing: -.035em;
    }
    .hero p { margin: 0; color: #a9bdc9; font-size: .94rem; }
    .status {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        margin: .35rem 0 1rem;
        padding: .38rem .75rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(11, 28, 41, .7);
        color: #b8cbd6;
        font-size: .78rem;
        font-weight: 650;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #42e0c1;
        box-shadow: 0 0 12px rgba(66, 224, 193, .8);
    }
    .dot.stale { background: #f4b860; box-shadow: 0 0 12px rgba(244, 184, 96, .65); }
    .dot.waiting { background: #8295a4; box-shadow: none; }
    [data-testid="stMetric"] {
        min-height: 122px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(18, 39, 54, .94), var(--panel));
        box-shadow: 0 10px 24px rgba(0, 0, 0, .12);
    }
    [data-testid="stMetricLabel"] { color: #9db3c1 !important; }
    [data-testid="stMetricValue"] { color: #edf8fa !important; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    h2, h3 { color: #e6f0f4 !important; letter-spacing: -.02em; }
    div[data-testid="stTabs"] button { color: #9db3c1; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal);
        border-bottom-color: var(--teal);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_latest():
    if not os.path.exists(LATEST_DATA_PATH):
        return None
    with open(LATEST_DATA_PATH, encoding="utf-8") as source:
        return json.load(source)


def load_log():
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_LOG_PATH)


def numeric_value(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def freshness(processed_at):
    if not processed_at:
        return "Waiting for subscriber data", "waiting"
    try:
        timestamp = datetime.fromisoformat(processed_at)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_seconds = max(0, (datetime.now(timezone.utc) - timestamp).total_seconds())
    except (TypeError, ValueError):
        return "Last update time unavailable", "stale"

    if age_seconds < 30:
        return f"Live · updated {int(age_seconds)}s ago", "live"
    return f"Stale · last update {int(age_seconds / 60)}m ago", "stale"


def render():
    latest = load_latest()
    log = load_log()
    prediction_column = "predicted_% Silica Concentrate"
    actual_target = "% Silica Concentrate"

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">P_220 · Industrial process intelligence</div>
          <h1>Concentrate Quality Monitor</h1>
          <p>Real-time flotation process signals and XGBoost silica predictions</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if latest is None or prediction_column not in latest:
        st.markdown(
            '<div class="status"><span class="dot waiting"></span>'
            "Waiting for mining-process predictions</div>",
            unsafe_allow_html=True,
        )
        st.info("Start `mqtt_subscriber.py` and `sensor_simulator.py` to stream data.")
        return

    status, status_class = freshness(latest.get("processed_at"))
    st.markdown(
        f'<div class="status"><span class="dot {status_class}"></span>'
        f"{status} &nbsp;·&nbsp; Batch {latest.get('batch_id', '—')}</div>",
        unsafe_allow_html=True,
    )

    prediction = numeric_value(latest.get(prediction_column))
    actual = numeric_value(latest.get(actual_target))
    error = abs(prediction - actual) if prediction is not None and actual is not None else None
    updated = latest.get("processed_at", "")
    if updated:
        try:
            updated = datetime.fromisoformat(updated).strftime("%H:%M:%S UTC")
        except ValueError:
            pass

    metric_columns = st.columns(4)
    metric_columns[0].metric("Dataset batch", latest.get("batch_id", "—"))
    metric_columns[1].metric(
        "Predicted silica concentrate",
        f"{prediction:.4f}%" if prediction is not None else "—",
    )
    metric_columns[2].metric(
        "Actual silica concentrate",
        f"{actual:.4f}%" if actual is not None else "—",
    )
    metric_columns[3].metric(
        "Absolute prediction error",
        f"{error:.4f}%" if error is not None else "—",
        help="Absolute difference between the model prediction and measured target.",
    )

    overview_tab, signals_tab, prediction_tab, history_tab = st.tabs(
        ["Process overview", "Sensor analytics", "Quality analytics", "Recent records"]
    )

    with overview_tab:
        st.subheader("From ore feed to quality prediction")
        st.markdown(
            """
            <svg viewBox="0 0 1120 190" role="img"
                 aria-label="Mining ore feed flows through flotation and sensor monitoring to a quality prediction"
                 style="width:100%;height:auto;display:block;border:1px solid rgba(149,184,202,.16);
                        border-radius:14px;background:linear-gradient(110deg,#0c1d2a,#102d38);">
              <defs>
                <linearGradient id="pipe" x1="0" x2="1">
                  <stop offset="0" stop-color="#378f9b"/><stop offset="1" stop-color="#42e0c1"/>
                </linearGradient>
              </defs>
              <g fill="none" stroke="url(#pipe)" stroke-width="3">
                <path d="M212 94h125"/><path d="M490 94h125"/><path d="M770 94h125"/>
              </g>
              <g fill="#42e0c1">
                <path d="m330 87 13 7-13 7z"/><path d="m608 87 13 7-13 7z"/>
                <path d="m888 87 13 7-13 7z"/>
              </g>
              <g font-family="Segoe UI, sans-serif" text-anchor="middle">
                <g transform="translate(110 73)" stroke="#88aabb" stroke-width="3" fill="none">
                  <path d="M-40 4 0-22 40 4M-30 4v30h60V4M-8 34V13H8v21"/>
                  <path d="M-48 42h96"/>
                </g>
                <text x="110" y="145" fill="#e8f0f6" font-size="15" font-weight="700">ORE FEED</text>
                <text x="110" y="165" fill="#91a6b8" font-size="12">Mining process readings</text>
                <g transform="translate(390 71)" stroke="#42e0c1" stroke-width="3" fill="none">
                  <path d="M-38 0h76l-8 57h-60zM-30 14h60M-22 28h44"/>
                  <path d="M-24-11v11M0-11v11M24-11v11"/>
                  <circle cx="-17" cy="42" r="3" fill="#42e0c1"/>
                  <circle cx="0" cy="42" r="3" fill="#42e0c1"/>
                  <circle cx="17" cy="42" r="3" fill="#42e0c1"/>
                </g>
                <text x="390" y="145" fill="#e8f0f6" font-size="15" font-weight="700">FLOTATION</text>
                <text x="390" y="165" fill="#91a6b8" font-size="12">Air, pulp, pH &amp; flow signals</text>
                <g transform="translate(670 75)" stroke="#88aabb" stroke-width="3" fill="none">
                  <rect x="-38" y="-18" width="76" height="54" rx="8"/>
                  <path d="M-22 18V6l13-12 12 17 9-9 11 8"/>
                  <circle cx="-21" cy="-6" r="3" fill="#42e0c1" stroke="none"/>
                </g>
                <text x="670" y="145" fill="#e8f0f6" font-size="15" font-weight="700">SENSOR STREAM</text>
                <text x="670" y="165" fill="#91a6b8" font-size="12">MQTT → model inputs</text>
                <g transform="translate(950 73)" stroke="#42e0c1" stroke-width="3" fill="none">
                  <path d="M0-28 34-15v24c0 22-15 36-34 45C-19 45-34 31-34 9v-24z"/>
                  <path d="m-15 7 10 10 21-23"/>
                </g>
                <text x="950" y="145" fill="#e8f0f6" font-size="15" font-weight="700">QUALITY</text>
                <text x="950" y="165" fill="#91a6b8" font-size="12">XGBoost silica estimate</text>
              </g>
            </svg>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The simulator publishes each complete dataset row over MQTT. "
            "The subscriber validates the sensor inputs, predicts silica "
            "concentrate, then writes the latest result and prediction history."
        )
        overview_left, overview_right = st.columns(2)
        overview_left.metric(
            "Process inputs in this reading",
            sum(
                1
                for key, value in latest.items()
                if key not in {
                    "date",
                    "batch_id",
                    "% Iron Concentrate",
                    "% Silica Concentrate",
                    prediction_column,
                    "processed_at",
                }
                and numeric_value(value) is not None
            ),
        )
        overview_right.metric("Logged predictions", f"{len(log):,}")

    with signals_tab:
        st.subheader("Live process measurements")
        feature_columns = [
            column
            for column, value in latest.items()
            if column not in {
                "date",
                "batch_id",
                "% Iron Concentrate",
                "% Silica Concentrate",
                prediction_column,
                "processed_at",
            }
            and numeric_value(value) is not None
        ]
        if feature_columns:
            selected_features = st.multiselect(
                "Select signals",
                feature_columns,
                default=feature_columns[: min(4, len(feature_columns))],
            )
            signal_columns = st.columns(4)
            for index, feature in enumerate(selected_features):
                value = numeric_value(latest[feature])
                signal_columns[index % 4].metric(feature, f"{value:,.3f}")

            chart_features = [
                feature for feature in selected_features if feature in log.columns
            ]
            if chart_features and not log.empty:
                history = log[chart_features].apply(pd.to_numeric, errors="coerce")
                history = history.replace([float("inf"), float("-inf")], pd.NA)
                history = history.dropna(how="all").tail(200).reset_index(drop=True)
                if not history.empty:
                    st.caption(
                        "Selected signal history · each sensor scaled to 0–1 "
                        "to make different units comparable"
                    )
                    minima = history.min()
                    spans = (history.max() - minima).replace(0, 1)
                    normalized = (history - minima) / spans
                    st.line_chart(normalized, height=300)

                    distribution_feature = st.selectbox(
                        "Inspect reading distribution",
                        chart_features,
                    )
                    values = pd.to_numeric(
                        log[distribution_feature], errors="coerce"
                    ).dropna()
                    if not values.empty:
                        bins = pd.cut(values, bins=min(15, max(2, values.nunique())))
                        distribution = (
                            bins.value_counts(sort=False)
                            .rename_axis("Sensor-value range")
                            .to_frame("Readings")
                        )
                        distribution.index = distribution.index.map(str)
                        st.bar_chart(distribution, height=260)
                        st.caption(
                            f"Distribution of {len(values):,} logged readings for "
                            f"{distribution_feature}."
                        )
                    else:
                        st.info(
                            f"No historical values are available for {distribution_feature} yet."
                        )
            else:
                st.info(
                    "Selected signals will chart after matching measurements "
                    "have been logged."
                )
        else:
            st.info("No numeric process signals are available in the latest reading.")

    with prediction_tab:
        st.subheader("Silica concentrate: actual vs predicted")
        if not log.empty and prediction_column in log.columns:
            trend_columns = [prediction_column]
            if actual_target in log.columns:
                trend_columns.append(actual_target)
            trend = log[trend_columns].apply(pd.to_numeric, errors="coerce")
            trend = trend.replace([float("inf"), float("-inf")], pd.NA)
            trend = trend.dropna(how="all").tail(500).reset_index(drop=True)
            if not trend.empty:
                quality_left, quality_right = st.columns(2)
                with quality_left:
                    st.caption("Latest 500 readings · measured vs model output")
                    st.line_chart(trend, height=320)
                paired = trend.dropna(
                    subset=[prediction_column, actual_target]
                    if actual_target in trend.columns
                    else [prediction_column]
                )
                with quality_right:
                    st.caption("Prediction agreement · points nearer the diagonal are closer")
                    if actual_target in paired.columns and len(paired) > 1:
                        st.scatter_chart(
                            paired,
                            x=actual_target,
                            y=prediction_column,
                            height=320,
                        )
                    else:
                        st.info(
                            "Actual and predicted pairs are needed for the agreement chart."
                        )

                if actual_target in trend.columns:
                    error_history = trend.dropna(
                        subset=[prediction_column, actual_target]
                    ).copy()
                    if not error_history.empty:
                        error_history["Absolute error (%)"] = (
                            error_history[prediction_column]
                            - error_history[actual_target]
                        ).abs()
                        st.subheader("Prediction error by reading")
                        st.area_chart(
                            error_history[["Absolute error (%)"]].reset_index(drop=True),
                            height=240,
                        )
                        st.caption(
                            "Absolute difference between predicted and measured "
                            "silica concentrate; lower values indicate closer predictions."
                        )
                st.caption("Charts use only rows with available numeric values.")
            else:
                st.info("Prediction history is waiting for valid readings.")
        else:
            st.info("Prediction history will appear after the first reading is logged.")

    with history_tab:
        st.subheader("Latest prediction records")
        if log.empty:
            st.info("No prediction records have been logged yet.")
        else:
            st.dataframe(log.tail(100), width="stretch", hide_index=True)
            st.caption(f"Showing the latest 100 of {len(log):,} logged records.")

    st.caption(f"Last processed: {updated or 'not available'}")


st.sidebar.markdown("### Control room")
st.sidebar.caption("P_220 · Flotation quality monitoring")
refresh = st.sidebar.slider("Refresh interval", 1, 10, 2, format="%d sec")
auto_refresh = st.sidebar.toggle("Live auto-refresh", value=True)
if st.sidebar.button("Refresh now", use_container_width=True):
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Data source")
st.sidebar.code("MQTT → XGBoost → dashboard", language=None)

st.fragment(run_every=refresh if auto_refresh else None)(render)()
