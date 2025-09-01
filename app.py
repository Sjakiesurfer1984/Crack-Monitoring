import os
import shutil
import time
import logging
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict

import apsw
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database_handler import DatabaseHandler, DB_PATH
from data_loader import DataLoader

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a fixed palette of 10 dark colors

PALETTE: List[str] = [
    "#005eff", "#5500FF", '#ff7c43', '#ffa600', 
    "#00FF00", "#FF0000", "#ff00bf", "#00D9FF", 
    "#333333", "#007054", 
]


def get_plotly_layout() -> Dict:
    """
    Always returns a light-theme Plotly layout for charts.
    """
    font_color: str = '#111111'
    return {
        'template': 'plotly',
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': dict(color=font_color),
        'xaxis': dict(tickfont=dict(color=font_color), title=dict(font=dict(color=font_color))),
        'yaxis': dict(tickfont=dict(color=font_color), title=dict(font=dict(color=font_color))),
        'legend': dict(font=dict(color=font_color)),
        'legend_title': dict(font=dict(color=font_color)),
        'hovermode': 'x unified'
    }


def compute_y_range(
    series: pd.Series,
    padding_factor: float = 0.2,
    zero_centered_factor: float = 1.2
) -> Tuple[float, float]:
    y_min, y_max = series.min(), series.max()
    if y_min < 0 < y_max:
        max_var = max(abs(y_min), abs(y_max))
        pad = max_var * zero_centered_factor
        return -pad, pad
    span = y_max - y_min
    pad = span * padding_factor if span != 0 else zero_centered_factor * 0.05
    return y_min - pad, y_max + pad


# def create_line_figure(
#     df: pd.DataFrame,
#     x_col: str,
#     y_col: str,
#     y_label: str,
#     color: str,
#     title_text: str,
#     y_range: Tuple[float, float],
#     tick_format: str = ".3f",
#     font_sizes: Optional[Dict[str, int]] = None
# ) -> go.Figure:
#     fig = px.line(
#         df,
#         x=x_col,
#         y=y_col,
#         title=title_text,
#         labels={x_col: 'Date & Time', y_col: y_label},
#         template='plotly',
#         color_discrete_sequence=PALETTE  # ensure no yellow
#     )
#     fig.update_traces(name=y_col, showlegend=True, line=dict(color=color))
#     # fig.update_yaxes(range=y_range, tickformat=tick_format)
#     fig.update_yaxes(
#     range=y_range, 
#     tickformat=tick_format,
#     dtick=0.2,       # force a tick every 0.2 units on the Y-axis
#     showgrid=True,   # turn on the horizontal grid lines
#     gridcolor='lightgrey',  # pick a subtle grey for the grid
#     gridwidth=0.5      # (optional) control line thickness
#     )
#     fig.update_xaxes(
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=0.5
#     )
#     defaults = {'base':14,'legend':16,'axis_title':16,'axis_tick':14,'title':18}
#     fonts = {**defaults, **(font_sizes or {})}
#     fig.update_layout(
#         font=dict(size=fonts['base'], color='#111111'),
#         legend=dict(font=dict(size=fonts['legend'])),
#         xaxis=dict(title=dict(font=dict(size=fonts['axis_title'])), tickfont=dict(size=fonts['axis_tick'])),
#         yaxis=dict(title=dict(font=dict(size=fonts['axis_title'])), tickfont=dict(size=fonts['axis_tick'])),
#         title=dict(
#             text=title_text, x=0.5, xanchor='center', yanchor='top', y=0.95,
#             font=dict(size=fonts['title'], color='#111111')
#         ),
#         margin=dict(t=100, r = 240)
#     )
#     return fig


def create_line_figure(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_label: str,
    color: str,
    title_text: str,
    y_range: Tuple[float, float],
    tick_format: str = ".3f",
    font_sizes: Optional[Dict[str, int]] = None
) -> go.Figure:
    """
    Build a light-themed Plotly line figure with fixed y-axis and full grid.
    """
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title_text,
        labels={x_col: 'Date & Time', y_col: y_label},
        template='plotly',
        color_discrete_sequence=PALETTE
    )

    # Trace styling
    fig.update_traces(
        name=y_col,
        showlegend=True,
        line=dict(color=color)
    )

    # Y-axis ticks & grid
    fig.update_yaxes(
        range=y_range,
        tickformat=tick_format,
        dtick=0.2,
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1
    )
    # X-axis grid lines
    fig.update_xaxes(
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1
    )

    # Font sizes default or overrides
    defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
    fonts = {**defaults, **(font_sizes or {})}

        # Layout including legend positioned outside
    fig.update_layout(
        font=dict(size=fonts['base'], color='#111111'),
        legend=dict(
            font=dict(size=fonts['legend']),
            orientation='v',      # vertical legend
            x=1.02,               # move just outside plot area
            xanchor='left',
            y=1,
            yanchor='top'
        ),
        xaxis=dict(
            title=dict(font=dict(size=fonts['axis_title'])),
            tickfont=dict(size=fonts['axis_tick'])
        ),
        yaxis=dict(
            title=dict(font=dict(size=fonts['axis_title'])),
            tickfont=dict(size=fonts['axis_tick'])
        ),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            yanchor='top',
            y=0.95,
            font=dict(size=fonts['title'], color='#111111')
        ),
        margin=dict(t=100, r=300)  # increase right margin
    )

    return fig

def render_individual_graphs(
    df_filtered: pd.DataFrame,
    selected_channels: List[str],
    start_date: date,
    end_date: date
) -> None:
    """
    Render each channel in its own light-themed figure with fixed y-axis range.
    """
    st.subheader("📉 Individual Channel Graphs")

    for idx, channel in enumerate(selected_channels):
        st.markdown(f"### {channel}")
        if channel not in df_filtered.columns:
            st.warning(f"⚠️ No data for {channel}")
            continue

        # Prepare data
        df_ch = df_filtered[['datetime', channel]].dropna()
        df_ch[channel] = pd.to_numeric(df_ch[channel], errors='coerce').dropna()
        if df_ch.empty:
            st.warning(f"⚠️ No valid numeric data for {channel}")
            continue

        # Fixed y-axis range (from -0.800 to +1.600)
        # This is the hard-coded line setting the scale:
        y_range = (-0.8, 1.6)

        # Figure parameters
        color = PALETTE[idx % len(PALETTE)]
        title_text = f"{channel}: {start_date} to {end_date}"
        y_label = 'Crack width change [mm]'

        # Build and display figure
        fig = create_line_figure(
            df=df_ch,
            x_col='datetime',
            y_col=channel,
            y_label=y_label,
            color=color,
            title_text=title_text,
            y_range=y_range
        )
        fig.update_layout(**get_plotly_layout())
        st.plotly_chart(fig, use_container_width=True)

# def render_combined_normalised_graph(
#     df: pd.DataFrame,
#     all_channel_cols: List[str],
#     start_date_default: date,
#     end_date_default: date
# ) -> None:
#     """
#     Render all channels in one light-themed figure with:
#     - fixed y-axis from -0.8 to +1.6
#     - 0.2 unit tick spacing
#     - both horizontal & vertical grid lines
#     - consistent font sizes & margins
#     """
#     st.subheader("📈 Combined Normalised Graph")
#     if df.empty:
#         st.warning("📪 No records for selected range")
#         return

#     # Melt to long form
#     df_m = df.melt(
#         id_vars='datetime',
#         value_vars=all_channel_cols,
#         var_name='Sensor',
#         value_name='Value'
#     )
#     if df_m.empty:
#         st.warning("⚠️ Nothing to plot.")
#         return

#     title_text = f"Sensor Data from {start_date_default} to {end_date_default}"
#     y_range = (-0.8, 1.6)

#     # Build the multi-line figure
#     fig: go.Figure = px.line(
#         df_m,
#         x='datetime',
#         y='Value',
#         color='Sensor',
#         title=title_text,
#         labels={'datetime': 'Date & Time', 'Value': 'Crack width change [mm]'},
#         template='plotly',
#         color_discrete_sequence=PALETTE
#     )

#     # Y-axis ticks & grid
#     fig.update_yaxes(
#         range=y_range,
#         tickformat=".3f",
#         dtick=0.2,
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=1
#     )
#     # X-axis grid
#     fig.update_xaxes(
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=1
#     )

#     # Match the same font-size scheme as individual plots
#     defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
#     fig.update_layout(
#         font=dict(size=defaults['base'], color='#111111'),
#         legend=dict(font=dict(size=defaults['legend'])),
#         xaxis=dict(title=dict(font=dict(size=defaults['axis_title'])), tickfont=dict(size=defaults['axis_tick'])),
#         yaxis=dict(title=dict(font=dict(size=defaults['axis_title'])), tickfont=dict(size=defaults['axis_tick'])),
#         title=dict(
#             text=title_text,
#             x=0.5, xanchor='center', yanchor='top', y=0.95,
#             font=dict(size=defaults['title'], color='#111111')
#         ),
#         margin=dict(t=60)
#     )

#     # Apply the same consistent light layout
#     fig.update_layout(**get_plotly_layout())

#     st.plotly_chart(fig, use_container_width=True)



def render_combined_normalised_graph(
    df: pd.DataFrame,
    all_channel_cols: List[str],
    start_date_default: date,
    end_date_default: date
) -> None:
    """
    Render all channels in one light-themed figure with fixed y-axis range, tick granularity,
    full horizontal & vertical grid, consistent font sizes, and legend positioned outside.
    """
    st.subheader("📈 Combined Normalised Graph")
    if df.empty:
        st.warning("📪 No records for selected range")
        return

    # Melt DataFrame
    df_m = df.melt(
        id_vars='datetime',
        value_vars=all_channel_cols,
        var_name='Sensor',
        value_name='Value'
    )
    if df_m.empty:
        st.warning("⚠️ Nothing to plot.")
        return

    title_text = f"Sensor Data from {start_date_default} to {end_date_default}"
    y_range: Tuple[float, float] = (-0.8, 1.6)

    fig: go.Figure = px.line(
        df_m,
        x='datetime',
        y='Value',
        color='Sensor',
        title=title_text,
        labels={'datetime': 'Date & Time', 'Value': 'Crack width change [mm]'},
        template='plotly',
        color_discrete_sequence=PALETTE
    )

    fig.update_yaxes(
        range=y_range,
        tickformat=".3f",
        dtick=0.2,
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1
    )

    defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
    fig.update_layout(
        font=dict(size=defaults['base'], color='#111111'),
        legend=dict(
            font=dict(size=defaults['legend']),
            orientation='v',
            x=1.02,
            xanchor='left',
            y=1,
            yanchor='top'
        ),
        xaxis=dict(title=dict(font=dict(size=defaults['axis_title'])), tickfont=dict(size=defaults['axis_tick'])),
        yaxis=dict(title=dict(font=dict(size=defaults['axis_title'])), tickfont=dict(size=defaults['axis_tick'])),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            yanchor='top',
            y=0.95,
            font=dict(size=defaults['title'], color='#111111')
        ),
        margin=dict(t=60, r=300)
    )
    fig.update_layout(**get_plotly_layout())

    st.plotly_chart(fig, use_container_width=True)





def render_statistics(
    df: pd.DataFrame,
    columns: List[str]
) -> None:
    """
    Render stats and histogram for selected channel.
    """
    st.subheader("📊 Channel Statistics")
    with st.form("stats_form"):
        sel = st.selectbox("Select a channel:", options=columns)
        btn = st.form_submit_button("Generate Stats")
    if btn and sel:
        vals = pd.to_numeric(df[sel], errors='coerce').dropna()
        if vals.empty:
            st.warning("⚠️ No numeric data.")
            return
        mean, std, vmin, vmax = vals.mean(), vals.std(), vals.min(), vals.max()
        # Noting that the mean is the mean crack width DIFFERENCE wrt to the initial gap width at start of the measurements (i.e. 17th of January 2024 for most sensors). 
        # Second note: The gap width difference can also become negative. Hence, the mean may not be a useful metric?
        st.markdown(f"**Mean:** {mean:.3f}")
        st.markdown(f"**Std Dev:** {std:.3f}")
        st.markdown(f"**Min:** {vmin:.3f}")
        st.markdown(f"**Max:** {vmax:.3f}")
        hist = go.Figure(data=[go.Histogram(x=vals)])
        hist.update_layout(
            **get_plotly_layout(),
            title=f"Distribution of {sel}", xaxis_title=sel, yaxis_title="Count"
        )
        st.plotly_chart(hist, use_container_width=True)


def backup_database(
    db_path: str = DB_PATH,
    backup_dir: str = "backups"
) -> Optional[str]:
    """
    Backup DB if it has tables/data.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB missing: {db_path}")
    conn = apsw.Connection(db_path); cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    conn.close()
    if not tables:
        logger.info("No tables; skip backup.")
        return None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(os.path.dirname(__file__), backup_dir)
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, f"backup_{ts}.db")
    shutil.copy2(db_path, dst)
    logger.info(f"Backed up to {dst}")
    return dst



st.title("\U0001F4CA Multi-Sensor Data Explorer")
st.session_state["theme"] = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

db = DatabaseHandler()
df_all = db.load_all_data()

if not df_all.empty:
    df_all = df_all.sort_values("datetime")
    all_channel_cols = [col for col in df_all.columns if col.endswith("profile")]

    # Initialize session state defaults
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = df_all["datetime"].min().date()
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = df_all["datetime"].max().date()
    if "selected_channels" not in st.session_state:
        st.session_state["selected_channels"] = all_channel_cols.copy()

    with st.form("filter_form"):
        start_date = st.date_input("Start Date", st.session_state["start_date"])
        end_date = st.date_input("End Date", st.session_state["end_date"])
        selected_channels = st.multiselect(
            "Select Channels",
            options=all_channel_cols,
            default=st.session_state["selected_channels"],
        )
        if st.form_submit_button("Apply Filters"):
            st.session_state["start_date"] = start_date
            st.session_state["end_date"] = end_date
            st.session_state["selected_channels"] = selected_channels

    # Now work with the persisted state
    if not st.session_state["selected_channels"]:
        st.warning("\u26A0\uFE0F No channels selected. Defaulting to all available.")
        st.session_state["selected_channels"] = all_channel_cols.copy()

    df_filtered = df_all[
        (df_all['datetime'].dt.date >= st.session_state["start_date"]) &
        (df_all['datetime'].dt.date <= st.session_state["end_date"])
    ]

    if df_filtered.empty:
        st.warning("\U0001F4EC No records for selected range")
    else:
        render_combined_normalised_graph(
            df_filtered,
            st.session_state["selected_channels"],
            st.session_state["start_date"],
            st.session_state["end_date"]
        )
        render_individual_graphs(
            df_filtered,
            st.session_state["selected_channels"],
            st.session_state["start_date"],
            st.session_state["end_date"],
        )
        render_statistics(df_filtered, st.session_state["selected_channels"])

# ----------------- START APP ----------------------------------------------
st.markdown("---")
st.subheader("📂 Upload New Sensor File")

# ← only one uploader in the whole file
uploaded_file = st.file_uploader(
    "Upload Excel or CSV File",
    type=["csv", "xlsx", "xls"],
    key="uploader"
)
if uploaded_file is not None:
    fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"

    # only process each file once
    if st.session_state.get("last_upload") == fingerprint:
        st.info("⚠️ This file was already processed.")
    else:
        # Step 1: Backup
        t0 = time.time()
        st.write("▶️ Step 1: Backing up database…")
        backup_database()
        st.write(f"✅ Backup done in {time.time() - t0:.2f}s")

        # Step 2: Load & Clean
        t1 = time.time()
        st.write("▶️ Step 2: Loading & cleaning file…")
        loader = DataLoader(uploaded_file)
        df_new = loader.load_and_clean()
        print(f"Loaded {len(df_new)} rows from uploaded file")
        st.write(f"✅ load_and_clean done in {time.time() - t1:.2f}s (shape={df_new.shape})")

        # Step 3: Compute Diffs
        t2 = time.time()
        st.write("▶️ Step 3: Computing channel diffs…")
        calibration_factors = [
            0.002850975, 0.002861057, 0.002860953, 0.002837607,
            0.002918660, 0.002953280, 0.002905340, 0.002928900
        ]
        df_new = loader.compute_channel_differences(df_new, calibration_factors)
        st.write(f"✅ compute_channel_differences done in {time.time() - t2:.2f}s")

        # Step 4: Query Latest Datetime
        t3 = time.time()
        st.write("▶️ Step 4: Querying latest datetime from DB…")
        latest_dt = db.get_latest_datetime()
        st.write(f"✅ get_latest_datetime returned {latest_dt!r} in {time.time() - t3:.2f}s")

        # Step 5: Filter & Save
        t4 = time.time()
        st.write("▶️ Step 5: Filtering new rows & saving to DB…")
        if latest_dt is not None:
            df_new = df_new[df_new["datetime"] > latest_dt]
            st.write(f"   • {len(df_new)} rows newer than {latest_dt}")
        if df_new.empty:
            st.warning("No new rows to insert.")
        else:
            db.save_to_db(df_new)
            st.write(f"✅ save_to_db done in {time.time() - t4:.2f}s (inserted {len(df_new)} rows)")

        # Remember we processed this file
        st.session_state["last_upload"] = fingerprint
        st.success("🎉 Upload sequence complete.")

else:
    last = st.session_state.get("last_upload")
    if last:
        st.info(f"Last uploaded file: {last}")
    else:
        st.info("⚠️ No file uploaded yet. Please select one above.")