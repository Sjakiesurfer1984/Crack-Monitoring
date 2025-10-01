# import os
# import shutil
# import time
# import logging
# from datetime import datetime, date
# from typing import List, Tuple, Optional, Dict

# import apsw
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# import math

# from database_handler import DatabaseHandler, DB_PATH
# from data_loader import DataLoader

# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define a fixed palette of 10 dark colors

# PALETTE: List[str] = [
#     "#005eff", "#5500FF", '#ff7c43', '#ffa600', 
#     "#00FF00", "#FF0000", "#ff00bf", "#00D9FF", 
#     "#333333", "#007054", 
# ]


# def get_plotly_layout() -> Dict:
#     """
#     Always returns a light-theme Plotly layout for charts.
#     """
#     font_color: str = '#111111'
#     return {
#         'template': 'plotly',
#         'plot_bgcolor': 'white',
#         'paper_bgcolor': 'white',
#         'font': dict(color=font_color),
#         'xaxis': dict(tickfont=dict(color=font_color), title=dict(font=dict(color=font_color))),
#         'yaxis': dict(tickfont=dict(color=font_color), title=dict(font=dict(color=font_color))),
#         'legend': dict(font=dict(color=font_color)),
#         'legend_title': dict(font=dict(color=font_color)),
#         'hovermode': 'x unified'
#     }


# def compute_y_range(
#     series: pd.Series,
#     padding_factor: float = 0.2,
#     zero_centered_factor: float = 1.2
# ) -> Tuple[float, float]:
#     y_min, y_max = series.min(), series.max()
#     if y_min < 0 < y_max:
#         max_var = max(abs(y_min), abs(y_max))
#         pad = max_var * zero_centered_factor
#         return -pad, pad
#     span = y_max - y_min
#     pad = span * padding_factor if span != 0 else zero_centered_factor * 0.05
#     return y_min - pad, y_max + pad


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
#     """
#     Build a light-themed Plotly line figure with fixed y-axis and full grid.
#     """
#     fig = px.line(
#         df,
#         x=x_col,
#         y=y_col,
#         title=title_text,
#         labels={x_col: 'Date & Time', y_col: y_label},
#         template='plotly',
#         color_discrete_sequence=PALETTE
#     )

#     # Trace styling
#     fig.update_traces(
#         name=y_col,
#         showlegend=True,
#         line=dict(color=color)
#     )

#     # Y-axis ticks & grid
#     fig.update_yaxes(
#         range=y_range,
#         tickformat=tick_format,
#         dtick=0.2,
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=1
#     )
#     # X-axis grid lines
#     fig.update_xaxes(
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=1
#     )

#     # Font sizes default or overrides
#     defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
#     fonts = {**defaults, **(font_sizes or {})}

#         # Layout including legend positioned outside
#     fig.update_layout(
#         font=dict(size=fonts['base'], color='#111111'),
#         legend=dict(
#             font=dict(size=fonts['legend']),
#             orientation='v',      # vertical legend
#             x=1.02,               # move just outside plot area
#             xanchor='left',
#             y=1,
#             yanchor='top'
#         ),
#         xaxis=dict(
#             title=dict(font=dict(size=fonts['axis_title'])),
#             tickfont=dict(size=fonts['axis_tick'])
#         ),
#         yaxis=dict(
#             title=dict(font=dict(size=fonts['axis_title'])),
#             tickfont=dict(size=fonts['axis_tick'])
#         ),
#         title=dict(
#             text=title_text,
#             x=0.5,
#             xanchor='center',
#             yanchor='top',
#             y=0.95,
#             font=dict(size=fonts['title'], color='#111111')
#         ),
#         margin=dict(t=100, r=300)  # increase right margin
#     )

#     return fig

# def calculate_flexible_y_range(series: pd.Series, padding_factor: float = 0.2) -> Tuple[float, float, Optional[float]]:
#     """
#     Calculates a dynamic, clean y-axis range and a suitable tick interval.

#     This function determines the min/max of the data, adds padding, rounds the
#     limits outward to a "nice" number, and computes a "nice" `dtick` interval
#     (e.g., 1, 2, 5, 10) that aims to produce between 4 and 10 ticks.

#     Args:
#         series (pd.Series): The pandas Series of numerical data for the y-axis.
#         padding_factor (float): The fractional padding to add to the data range.

#     Returns:
#         Tuple[float, float, Optional[float]]: A tuple of (rounded_min, rounded_max, nice_tick_interval).
#     """
#     data_min: float = series.min()
#     data_max: float = series.max()

#     if data_min == data_max:
#         padding = abs(data_min * padding_factor) if data_min != 0 else 0.1
#         # For a single point, dtick is not relevant, let Plotly decide.
#         return data_min - padding, data_max + padding, None

#     span: float = data_max - data_min
#     # Handle extremely small spans which might cause floating point issues
#     if span < 1e-9:
#         padding = abs(data_min * 0.1) if data_min != 0 else 0.1
#         return data_min - padding, data_max + padding, padding / 2

#     padding: float = span * padding_factor
#     padded_min: float = data_min - padding
#     padded_max: float = data_max + padding
#     padded_span: float = padded_max - padded_min

#     # --- Calculate a "nice" tick interval (dtick) ---
#     # Aim for a certain number of ticks (e.g., 5-10).
#     target_ticks = 8
#     raw_step = padded_span / target_ticks
    
#     # Calculate the order of magnitude of the raw step.
#     power = 10.0 ** math.floor(math.log10(raw_step))
    
#     # Normalize the raw step to be between 1 and 10.
#     normalized_step = raw_step / power
    
#     # Find the closest "nice" multiplier (1, 2, 5, or 10).
#     multiples = [1, 2, 2.5, 5, 10]
#     best_multiple = min(multiples, key=lambda x: abs(x - normalized_step))
    
#     final_dtick = best_multiple * power

#     # --- Round the padded min/max outward to a multiple of the nice tick interval ---
#     final_min = math.floor(padded_min / final_dtick) * final_dtick
#     final_max = math.ceil(padded_max / final_dtick) * final_dtick

#     return final_min, final_max, final_dtick

# def render_individual_graphs(
#     df_filtered: pd.DataFrame,
#     selected_channels: List[str],
#     start_date: date,
#     end_date: date
# ) -> None:
#     """
#     Render each channel in its own light-themed figure with fixed y-axis range.
#     """
#     st.subheader("📉 Individual Channel Graphs")

#     for idx, channel in enumerate(selected_channels):
#         st.markdown(f"### {channel}")
#         if channel not in df_filtered.columns:
#             st.warning(f"⚠️ No data for {channel}")
#             continue

#         # Prepare data
#         df_ch = df_filtered[['datetime', channel]].dropna()
#         df_ch[channel] = pd.to_numeric(df_ch[channel], errors='coerce').dropna()
#         if df_ch.empty:
#             st.warning(f"⚠️ No valid numeric data for {channel}")
#             continue
        

#         y_min, y_max, y_d_tick = calculate_flexible_y_range(df_ch[channel])

#         y_range = (y_min, y_max)
#         # Figure parameters
#         color = PALETTE[idx % len(PALETTE)]
#         title_text = f"{channel}: {start_date} to {end_date}"
#         y_label = 'Crack width change [mm]'

#         # Build and display figure
#         fig = create_line_figure(
#             df=df_ch,
#             x_col='datetime',
#             y_col=channel,
#             y_label=y_label,
#             color=color,
#             title_text=title_text,
#             y_range=y_range
#         )
#         fig.update_layout(**get_plotly_layout()) # Poorly coded, strong coupling. 
#         st.plotly_chart(fig, use_container_width=True)


# def render_combined_normalised_graph(
#     df: pd.DataFrame,
#     all_channel_cols: List[str],
#     start_date_default: date,
#     end_date_default: date
# ) -> None:
#     """
#     Render all channels in one light-themed figure with fixed y-axis range, tick granularity,
#     full horizontal & vertical grid, consistent font sizes, and legend positioned outside.
#     """
#     st.subheader("📈 Combined Normalised Graph")
#     if df.empty:
#         st.warning("📪 No records for selected range")
#         return

#     # Melt DataFrame
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
#     y_range: Tuple[float, float] = (-0.8, 1.6)

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

#     fig.update_yaxes(
#         range=y_range,
#         tickformat=".3f",
#         dtick=0.2,
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=1
#     )
#     fig.update_xaxes(
#         showgrid=True,
#         gridcolor='lightgrey',
#         gridwidth=1
#     )

#     defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
#     fig.update_layout(
#         font=dict(size=defaults['base'], color='#111111'),
#         legend=dict(
#             font=dict(size=defaults['legend']),
#             orientation='v',
#             x=1.02,
#             xanchor='left',
#             y=1,
#             yanchor='top'
#         ),
#         xaxis=dict(title=dict(font=dict(size=defaults['axis_title'])), tickfont=dict(size=defaults['axis_tick'])),
#         yaxis=dict(title=dict(font=dict(size=defaults['axis_title'])), tickfont=dict(size=defaults['axis_tick'])),
#         title=dict(
#             text=title_text,
#             x=0.5,
#             xanchor='center',
#             yanchor='top',
#             y=0.95,
#             font=dict(size=defaults['title'], color='#111111')
#         ),
#         margin=dict(t=60, r=300)
#     )
#     fig.update_layout(**get_plotly_layout())

#     st.plotly_chart(fig, use_container_width=True)


# def render_statistics(
#     df: pd.DataFrame,
#     columns: List[str]
# ) -> None:
#     """
#     Render stats and histogram for selected channel.
#     """
#     st.subheader("📊 Channel Statistics")
#     with st.form("stats_form"):
#         sel = st.selectbox("Select a channel:", options=columns)
#         btn = st.form_submit_button("Generate Stats")
#     if btn and sel:
#         vals = pd.to_numeric(df[sel], errors='coerce').dropna()
#         if vals.empty:
#             st.warning("⚠️ No numeric data.")
#             return
#         mean, std, vmin, vmax = vals.mean(), vals.std(), vals.min(), vals.max()
#         # Noting that the mean is the mean crack width DIFFERENCE wrt to the initial gap width at start of the measurements (i.e. 17th of January 2024 for most sensors). 
#         # Second note: The gap width difference can also become negative. Hence, the mean may not be a useful metric?
#         st.markdown(f"**Mean:** {mean:.3f}")
#         st.markdown(f"**Std Dev:** {std:.3f}")
#         st.markdown(f"**Min:** {vmin:.3f}")
#         st.markdown(f"**Max:** {vmax:.3f}")
#         hist = go.Figure(data=[go.Histogram(x=vals)])
#         hist.update_layout(
#             **get_plotly_layout(),
#             title=f"Distribution of {sel}", xaxis_title=sel, yaxis_title="Count"
#         )
#         st.plotly_chart(hist, use_container_width=True)


# def backup_database(
#     db_path: str = DB_PATH,
#     backup_dir: str = "backups"
# ) -> Optional[str]:
#     """
#     Backup DB if it has tables/data.
#     """
#     if not os.path.exists(db_path):
#         raise FileNotFoundError(f"DB missing: {db_path}")
#     conn = apsw.Connection(db_path); cur = conn.cursor()
#     tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';")]
#     conn.close()
#     if not tables:
#         logger.info("No tables; skip backup.")
#         return None
#     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#     out = os.path.join(os.path.dirname(__file__), backup_dir)
#     os.makedirs(out, exist_ok=True)
#     dst = os.path.join(out, f"backup_{ts}.db")
#     shutil.copy2(db_path, dst)
#     logger.info(f"Backed up to {dst}")
#     return dst


# # ----------------- START APP ----------------------------------------------

# st.title("\U0001F4CA Multi-Sensor Data Explorer")
# st.session_state["theme"] = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

# db = DatabaseHandler()
# df_all = db.load_all_data() # Load all data at once from the database. Note: this database is normally stored in the same folder as this script.
# # But, for development, I have it in a fixed path on the AIE share folder. I should change this before deployment, and make it configurable.
# # how? perhaps via an environment variable? Or a config file? Or a text input box in the sidebar? Or a file picker dialog?
# # a file picker dialog would be nice.

# if not df_all.empty:
#     df_all = df_all.sort_values("datetime")
#     all_channel_cols = [col for col in df_all.columns if col.endswith("profile")]

#     # Initialize session state defaults
#     if "start_date" not in st.session_state:
#         st.session_state["start_date"] = df_all["datetime"].min().date()
#     if "end_date" not in st.session_state:
#         st.session_state["end_date"] = df_all["datetime"].max().date()
#     if "selected_channels" not in st.session_state:
#         st.session_state["selected_channels"] = all_channel_cols.copy()

#     with st.form("filter_form"):
#         start_date = st.date_input("Start Date", st.session_state["start_date"])
#         end_date = st.date_input("End Date", st.session_state["end_date"])
#         selected_channels = st.multiselect(
#             "Select Channels",
#             options=all_channel_cols,
#             default=st.session_state["selected_channels"],
#         )
#         if st.form_submit_button("Apply Filters"):
#             st.session_state["start_date"] = start_date
#             st.session_state["end_date"] = end_date
#             st.session_state["selected_channels"] = selected_channels

#     # Now work with the persisted state
#     if not st.session_state["selected_channels"]:
#         st.warning("\u26A0\uFE0F No channels selected. Defaulting to all available.")
#         st.session_state["selected_channels"] = all_channel_cols.copy()

#     df_filtered = df_all[
#         (df_all['datetime'].dt.date >= st.session_state["start_date"]) &
#         (df_all['datetime'].dt.date <= st.session_state["end_date"])
#     ]

#     if df_filtered.empty:
#         st.warning("\U0001F4EC No records for selected range")
#     else:
#         render_combined_normalised_graph(
#             df_filtered,
#             st.session_state["selected_channels"],
#             st.session_state["start_date"],
#             st.session_state["end_date"]
#         )
#         render_individual_graphs(
#             df_filtered,
#             st.session_state["selected_channels"],
#             st.session_state["start_date"],
#             st.session_state["end_date"],
#         )
#         render_statistics(df_filtered, st.session_state["selected_channels"])


# st.markdown("---")
# st.subheader("📂 Upload New Sensor File")

# # ← only one uploader in the whole file
# uploaded_file = st.file_uploader(
#     "Upload Excel or CSV File",
#     type=["csv", "xlsx", "xls"],
#     key="uploader"
# )
# if uploaded_file is not None:
#     fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"

#     # only process each file once
#     if st.session_state.get("last_upload") == fingerprint:
#         st.info("⚠️ This file was already processed.")
#     else:
#         # Step 1: Backup
#         t0 = time.time()
#         st.write("▶️ Step 1: Backing up database…")
#         backup_database()
#         st.write(f"✅ Backup done in {time.time() - t0:.2f}s")

#         # Step 2: Load & Clean
#         t1 = time.time()
#         st.write("▶️ Step 2: Loading & cleaning file…")
#         loader = DataLoader(uploaded_file)
#         df_new = loader.load_and_clean()
#         print(f"Loaded {len(df_new)} rows from uploaded file")
#         st.write(f"✅ load_and_clean done in {time.time() - t1:.2f}s (shape={df_new.shape})")

#         # Step 3: Compute Diffs
#         t2 = time.time()
#         st.write("▶️ Step 3: Computing channel diffs…")
#         calibration_factors = [
#             0.002850975, 0.002861057, 0.002860953, 0.002837607,
#             0.002918660, 0.002953280, 0.002905340, 0.002928900
#         ]
#         df_new = loader.compute_channel_differences(df_new, calibration_factors)
#         st.write(f"✅ compute_channel_differences done in {time.time() - t2:.2f}s")

#         # Step 4: Query Latest Datetime
#         t3 = time.time()
#         st.write("▶️ Step 4: Querying latest datetime from DB…")
#         latest_dt = db.get_latest_datetime()
#         st.write(f"✅ get_latest_datetime returned {latest_dt!r} in {time.time() - t3:.2f}s")

#         # Step 5: Filter & Save
#         t4 = time.time()
#         st.write("▶️ Step 5: Filtering new rows & saving to DB…")
#         if latest_dt is not None:
#             df_new = df_new[df_new["datetime"] > latest_dt]
#             st.write(f"   • {len(df_new)} rows newer than {latest_dt}")
#         if df_new.empty:
#             st.warning("No new rows to insert.")
#         else:
#             db.save_to_db(df_new)
#             st.write(f"✅ save_to_db done in {time.time() - t4:.2f}s (inserted {len(df_new)} rows)")

#         # Remember we processed this file
#         st.session_state["last_upload"] = fingerprint
#         st.success("🎉 Upload sequence complete.")
#         # now auto reload all data by re-running the script
#         st.rerun()

# else:
#     last = st.session_state.get("last_upload")
#     if last:
#         st.info(f"Last uploaded file: {last}")
#     else:
#         st.info("⚠️ No file uploaded yet. Please select one above.")


import os
import shutil
import time
import logging
import math
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict, Any

import apsw
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Assuming these are custom modules in the same directory
# from database_handler import DatabaseHandler, DB_PATH
# from data_loader import DataLoader

# Mock classes for standalone execution if custom modules are not available
class MockDBHandler:
    def load_all_data(self):
        st.warning("Using mock data. `database_handler` not found.")
        # Check for numpy presence before using pd.np
        try:
            import numpy as np
        except ImportError:
            st.error("Numpy is required for mock data generation.")
            return pd.DataFrame()
        dates = pd.to_datetime(pd.date_range(start="2024-01-01", end="2024-03-31", freq='h'))
        data = {
            'datetime': dates
        }
        for i in range(8):
            # Create varied data series
            noise = (np.random.randn(len(dates)) * 0.1 * (i + 1)).cumsum()
            trend = np.linspace(start=0, stop=i*0.1, num=len(dates))
            data[f'CH{i+1}_profile'] = 0.5 + trend + noise
        return pd.DataFrame(data)
    def get_latest_datetime(self): return pd.to_datetime("2024-03-31")
    def save_to_db(self, df): pass
class MockDataLoader:
    def __init__(self, uploaded_file): pass
    def load_and_clean(self): return pd.DataFrame()
    def compute_channel_differences(self, df, factors): return df

# Use mock classes if the real ones can't be imported
try:
    from database_handler import DatabaseHandler, DB_PATH
    from data_loader import DataLoader
except ImportError:
    DatabaseHandler = MockDBHandler
    DataLoader = MockDataLoader
    DB_PATH = "mock.db"


# Logging setup
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

# Define a fixed palette of 10 dark colors
PALETTE: List[str] = [
    "#005eff", "#5500FF", '#ff7c43', '#ffa600',
    "#00FF00", "#FF0000", "#ff00bf", "#00D9FF",
    "#333333", "#007044",
]

# --- Plotting Utility Functions ---

def get_plotly_layout() -> Dict[str, Any]:
    """
    Always returns a base dictionary for a light-theme Plotly layout.
    """
    font_color: str = '#111111'
    return {
        'template': 'plotly',
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'color': font_color},
        'xaxis': {'tickfont': {'color': font_color}, 'title': {'font': {'color': font_color}}},
        'yaxis': {'tickfont': {'color': font_color}, 'title': {'font': {'color': font_color}}},
        'legend': {'font': {'color': font_color}, 'title': {'font': {'color': font_color}}},
        'hovermode': 'x unified'
    }


def calculate_flexible_y_range(series: pd.Series, padding_factor: float = 0.2) -> Tuple[float, float, Optional[float]]:
    """
    Calculates a dynamic, clean y-axis range and a suitable tick interval.
    """
    data_min: float = series.min()
    data_max: float = series.max()

    if data_min == data_max:
        padding = abs(data_min * padding_factor) if data_min != 0 else 0.1
        return data_min - padding, data_max + padding, None

    span: float = data_max - data_min
    if span < 1e-9:
        padding = abs(data_min * 0.1) if data_min != 0 else 0.1
        return data_min - padding, data_max + padding, padding / 2

    padding: float = span * padding_factor
    padded_min: float = data_min - padding
    padded_max: float = data_max + padding
    padded_span: float = padded_max - padded_min

    target_ticks = 8
    raw_step = padded_span / target_ticks
    power = 10.0 ** math.floor(math.log10(raw_step))
    normalized_step = raw_step / power
    multiples = [1, 2, 2.5, 5, 10]
    best_multiple = min(multiples, key=lambda x: abs(x - normalized_step))
    final_dtick = best_multiple * power

    final_min = math.floor(padded_min / final_dtick) * final_dtick
    final_max = math.ceil(padded_max / final_dtick) * final_dtick

    return final_min, final_max, final_dtick


def create_line_figure(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_label: str,
    color: str,
    title_text: str,
    y_range: Tuple[float, float],
    y_dtick: Optional[float],
    tick_format: str = ".3f",
    font_sizes: Optional[Dict[str, int]] = None
) -> go.Figure:
    """
    Build a self-contained, fully styled, light-themed Plotly line figure.
    This function now integrates the base theme internally to reduce coupling.
    """
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        labels={x_col: 'Date & Time', y_col: y_label},
        template='plotly',
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(
        name=y_col,
        showlegend=True,
        line=dict(color=color)
    )
    fig.update_yaxes(
        range=y_range,
        tickformat=tick_format,
        dtick=y_dtick,
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1
    )

    # --- Integrated Layout Configuration ---
    # Start with the base theme
    layout_config = get_plotly_layout()
    
    # Define font sizes, with defaults
    defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
    fonts = {**defaults, **(font_sizes or {})}
    
    # Merge specific font sizes into the base theme dictionary
    layout_config['font']['size'] = fonts['base']
    layout_config['legend']['font']['size'] = fonts['legend']
    layout_config['xaxis']['title']['font']['size'] = fonts['axis_title']
    layout_config['xaxis']['tickfont']['size'] = fonts['axis_tick']
    layout_config['yaxis']['title']['font']['size'] = fonts['axis_title']
    layout_config['yaxis']['tickfont']['size'] = fonts['axis_tick']

    # Add or override other layout properties
    layout_config['legend'].update(orientation='v', x=1.02, xanchor='left', y=1, yanchor='top')
    layout_config['title'] = {'text': title_text, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'y': 0.95, 'font': {'size': fonts['title'], 'color': '#111111'}}
    layout_config['margin'] = {'t': 100, 'r': 300}

    fig.update_layout(layout_config)
    
    return fig

# --- Streamlit Rendering Functions ---

def render_individual_graphs(
    df_filtered: pd.DataFrame,
    selected_channels: List[str],
    start_date: date,
    end_date: date
) -> None:
    """
    Render each channel in its own figure. The styling is now fully handled
    by the create_line_figure function.
    """
    st.subheader("📉 Individual Channel Graphs")

    for idx, channel in enumerate(selected_channels):
        st.markdown(f"### {channel}")
        if channel not in df_filtered.columns:
            st.warning(f"⚠️ No data for {channel}")
            continue

        df_ch = df_filtered[['datetime', channel]].dropna()
        df_ch[channel] = pd.to_numeric(df_ch[channel], errors='coerce')
        df_ch.dropna(subset=[channel], inplace=True)
        
        if df_ch.empty or df_ch[channel].isnull().all():
            st.warning(f"⚠️ No valid numeric data for {channel}")
            continue

        y_min, y_max, y_dtick = calculate_flexible_y_range(df_ch[channel])
        y_range = (y_min, y_max)

        color = PALETTE[idx % len(PALETTE)]
        title_text = f"{channel}: {start_date} to {end_date}"
        y_label = 'Crack width change [mm]'

        # Build and display figure. No extra styling call is needed here.
        fig = create_line_figure(
            df=df_ch,
            x_col='datetime',
            y_col=channel,
            y_label=y_label,
            color=color,
            title_text=title_text,
            y_range=y_range,
            y_dtick=y_dtick
        )
        st.plotly_chart(fig, use_container_width=True)


def render_combined_normalised_graph(
    df: pd.DataFrame,
    all_channel_cols: List[str],
    start_date_default: date,
    end_date_default: date
) -> None:
    """
    Render all channels in one figure, now with integrated styling logic.
    """
    st.subheader("📈 Combined Normalised Graph")
    if df.empty:
        st.warning("📪 No records for selected range")
        return

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

    # --- Integrated Layout Configuration ---
    layout_config = get_plotly_layout()
    defaults = {'base': 14, 'legend': 16, 'axis_title': 16, 'axis_tick': 14, 'title': 18}
    
    layout_config['font']['size'] = defaults['base']
    layout_config['legend']['font']['size'] = defaults['legend']
    layout_config['xaxis']['title']['font']['size'] = defaults['axis_title']
    layout_config['xaxis']['tickfont']['size'] = defaults['axis_tick']
    layout_config['yaxis']['title']['font']['size'] = defaults['axis_title']
    layout_config['yaxis']['tickfont']['size'] = defaults['axis_tick']

    layout_config['legend'].update(orientation='v', x=1.02, xanchor='left', y=1, yanchor='top')
    layout_config['title'] = dict(
        text=title_text,
        x=0.5,
        xanchor='center',
        yanchor='top',
        y=0.95,
        font=dict(size=defaults['title'], color='#111111')
    )
    layout_config['margin'] = dict(t=60, r=300)

    fig.update_layout(layout_config)
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
    if not os.path.exists(db_path) or db_path == "mock.db":
        logger.warning(f"DB missing or using mock at: {db_path}. Skipping backup.")
        return None
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


# ----------------- START APP ----------------------------------------------

st.title("📊 Multi-Sensor Data Explorer")
st.session_state["theme"] = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

db = DatabaseHandler()
df_all = db.load_all_data()

if df_all is not None and not df_all.empty:
    df_all = df_all.sort_values("datetime")
    all_channel_cols = [col for col in df_all.columns if col.endswith("profile")]

    if "start_date" not in st.session_state:
        st.session_state["start_date"] = df_all["datetime"].min().date()
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = df_all["datetime"].max().date()
    if "selected_channels" not in st.session_state:
        st.session_state["selected_channels"] = all_channel_cols[:]

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
            st.rerun()

    if not st.session_state["selected_channels"]:
        st.warning("⚠️ No channels selected. Please select at least one channel.")
    else:
        df_filtered = df_all[
            (df_all['datetime'].dt.date >= st.session_state["start_date"]) &
            (df_all['datetime'].dt.date <= st.session_state["end_date"])
        ]
        
        if df_filtered.empty:
            st.warning("📪 No records for selected range")
        else:
            selected_data = df_filtered[['datetime'] + st.session_state["selected_channels"]]
            render_combined_normalised_graph(
                selected_data,
                st.session_state["selected_channels"],
                st.session_state["start_date"],
                st.session_state["end_date"]
            )
            render_individual_graphs(
                selected_data,
                st.session_state["selected_channels"],
                st.session_state["start_date"],
                st.session_state["end_date"],
            )
            render_statistics(selected_data, st.session_state["selected_channels"])
else:
    st.error("Could not load any data. Please check the database connection.")


st.markdown("---")
st.subheader("📂 Upload New Sensor File")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV File",
    type=["csv", "xlsx", "xls"],
    key="uploader"
)
if uploaded_file is not None:
    fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_upload") == fingerprint:
        st.info("⚠️ This file was already processed.")
    else:
        st.write("▶️ Step 1: Backing up database…")
        t0 = time.time()
        backup_database()
        st.write(f"✅ Backup done in {time.time() - t0:.2f}s")

        st.write("▶️ Step 2: Loading & cleaning file…")
        t1 = time.time()
        loader = DataLoader(uploaded_file)
        df_new = loader.load_and_clean()
        st.write(f"✅ load_and_clean done in {time.time() - t1:.2f}s (shape={df_new.shape})")

        st.write("▶️ Step 3: Computing channel diffs…")
        t2 = time.time()
        calibration_factors = [
            0.002850975, 0.002861057, 0.002860953, 0.002837607,
            0.002918660, 0.002953280, 0.002905340, 0.002928900
        ]
        df_new = loader.compute_channel_differences(df_new, calibration_factors)
        st.write(f"✅ compute_channel_differences done in {time.time() - t2:.2f}s")

        st.write("▶️ Step 4: Querying latest datetime from DB…")
        t3 = time.time()
        latest_dt = db.get_latest_datetime()
        st.write(f"✅ get_latest_datetime returned {latest_dt!r} in {time.time() - t3:.2f}s")

        st.write("▶️ Step 5: Filtering new rows & saving to DB…")
        t4 = time.time()
        if latest_dt is not None:
            df_new = df_new[df_new["datetime"] > latest_dt]
            st.write(f"   • {len(df_new)} rows newer than {latest_dt}")
        if df_new.empty:
            st.warning("No new rows to insert.")
        else:
            db.save_to_db(df_new)
            st.write(f"✅ save_to_db done in {time.time() - t4:.2f}s (inserted {len(df_new)} rows)")

        st.session_state["last_upload"] = fingerprint
        st.success("🎉 Upload sequence complete.")
        st.rerun()

else:
    last = st.session_state.get("last_upload")
    if last:
        st.info(f"Last uploaded file: {last}")
    else:
        st.info("⚠️ No file uploaded yet. Please select one above.")

