import pandas as pd
import logging
from typing import Union, IO
import csv
from io import StringIO
import chardet
import re
import io
import numpy as np
# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    # `file_source` is expected to be a file-like object (i.e., implements IO interface: .read(), .name, etc.)
    # Example: this is what Streamlit's file_uploader returns

    def __init__(self, file_source: IO):
        """
        Constructor for the DataLoader class.

        Parameters:
        - file_source (IO): A file-like object (implements read(), seek(), name, etc.).
                            This can be a file uploaded via a UI (e.g., Streamlit's uploader).
                            It allows reading the content directly from memory (no need to save locally).
        """

        # Store the incoming file-like object as an instance variable
        # Data type: IO (generic input/output stream interface)
        self.file_source = file_source
        self.headers = []
        # Try to retrieve the name of the file from the file-like object
        # `getattr()` is used in case the object has no `.name` attribute — fallback = "Unknown Source"
        # `file_name` will be a string: the filename used for logging or diagnostics
        file_name = getattr(file_source, 'name', 'Unknown Source')

        # Log the fact that the DataLoader has been initialized and show the source filename
        logger.info(f"📅 DataLoader initialized with file: {file_name}")

    def compute_channel_differences(self, df: pd.DataFrame, calibration_factors: list[float]) -> pd.DataFrame:
        """
        Computes calibrated differences for all 'Channel X' columns against a fixed initial value.
        Adds new columns like 'Channel X crack profile' to the end of the DataFrame.
        """
        print("--- Compute Channel Differences ---")
        
        # Precisely select and sort channel columns using regex
        channel_cols = [col for col in df.columns if re.match(r"^Channel \d+$", col)]
        channel_cols.sort(key=lambda x: int(x.split()[-1]))
        
        print(f"Found and sorted Channel_cols: {channel_cols}")

        # Define the fixed initial (zero-reference) values for each channel
        initial_vals = [3823.6, 2606, 3438.1, np.nan, 2979.4, 3121.9, 2818.4, 3096.2]

        # --- Input Validation ---
        if len(channel_cols) != len(calibration_factors):
            raise ValueError(f"Mismatch: Found {len(channel_cols)} channels but received {len(calibration_factors)} calibration factors.")
        if len(channel_cols) != len(initial_vals):
            raise ValueError(f"Mismatch: Found {len(channel_cols)} channels but have {len(initial_vals)} initial values defined.")

        diff_columns = {}

        # Iterate through each channel column to calculate the differences
        for i, channel in enumerate(channel_cols):
            if channel not in df.columns:
                continue

            diff_col_name = f"{channel} crack profile"
            initial_value = initial_vals[i]
            diff_values = (df[channel] - initial_value) * calibration_factors[i]
            diff_columns[diff_col_name] = diff_values.round(6)

        # Append the new columns to the DataFrame
        df_with_diffs = pd.concat([df, pd.DataFrame(diff_columns)], axis=1)

        # ==================== DEBUG START (Top 20 Version) ====================
        # We loop through the entire diff_columns dataframe
        # to identify the top 20 largest values and print their values and row indices 
        # to the screen. This allows us to verify whether these large values are outliers and
        # spurious sensor values that need to be dealth with (e.g. forward fill)
        print("\n[DEBUG] Top 20 Max Value Analysis:")
        
        # Loop through the names of the columns we just created
        for col_name in diff_columns.keys():
            print(f"\n--- Top 20 for '{col_name}' ---")
            
            # Check if the column exists and has data
            if col_name in df_with_diffs and not df_with_diffs[col_name].isnull().all():
                
                # Use .nlargest(20) to get a Series of the top 20 values and their indices
                top_20_values = df_with_diffs[col_name].nlargest(20)
                
                # Check if any values were returned
                if top_20_values.empty:
                    print("  No data points found.")
                else:
                    # Iterate through the Series to print each value and its row index
                    for row_index, value in top_20_values.items():
                        print(f"  Row {row_index:<5}: {value:.6f}")
            else:
                print("  No valid data (all NaN).")
                
        print("-" * 60)
        # ===================== DEBUG END =====================
        return df_with_diffs
    

    def load_and_clean(self) -> pd.DataFrame:
        logger.info("🔄 Loading data...")
        df = None

        try:
            # ===============================================================
            #  LOAD THE RAW DATA BLOCK
            # ===============================================================
            if self.file_source.name.endswith((".xls", ".xlsx")):
                logger.info("--- Starting Excel load ---")
                df = pd.read_excel(self.file_source, sheet_name=0, skiprows=23, header=0)

            elif self.file_source.name.endswith(".csv"):
                logger.info("🔄 Starting CSV load and clean")

                # 1) rewind + read raw bytes → text buffer
                self.file_source.seek(0)
                raw = self.file_source.read()
                text = raw.decode("utf-8", errors="ignore")  # force utf-8, drop bad chars
                buf = io.StringIO(text)
                df = pd.read_csv(
                    buf,
                    sep=",",
                    skiprows=23,      # drop lines 1–23
                    header=0,         # line 24 becomes header
                    index_col=False,  # do not treat any column as index
                    engine="python"   # more forgiving parser
                )

            else:
                raise ValueError("Unsupported file type.")

            print("\n[STEP 1] RAW DATAFRAME LOADED FROM FILE:\n")
            print(df.head())
            print("-" * 60)

            # ===============================================================
            #  SHARED CLEANING PIPELINE WITH A PRINT AFTER EVERY STEP
            # ===============================================================
            
            # 3) drop the units row (now row 0 of df)
            df = df.iloc[1:].reset_index(drop=True)
            print("\n[2] After dropping units row:")
            print(df.head(5))

            # 4) clean up Unnamed / empty columns
            df = fix_unnamed_headers(df)
            print("\n[3] After fix_unnamed_headers:")
            print(df.head(5))

            # 5) coerce all except 'Date/time' to numeric
            for col in df.columns:
                if col != "Date/time":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            print("\n[4] After coercion to numeric:")
            print(df.head(5))

            # # 6) set 'Date/time' as index
            # df = df.set_index("Date/time")
            # print("\n[5] After setting 'Date/time' index:")
            # print(df.head(5))

            # 7) Renaming 'Date/time' to datetime
            df = df.rename(columns = {"Date/time" : "datetime"})
            print("\n[5] After renaming 'Date/time' to datetime:")
            print(df.head(5))
            # 8) Convert the column to actual datetime objects ✅ (<<< ADD THIS LINE HERE)
            df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
            
            logger.info(f"✅ Finished cleaning. Final shape: {df.shape}")
            print(df.head())
            return df

        except Exception as e:
            logger.error(f"❌ Failed to load or clean file: {e}")
            raise

def fix_unnamed_headers(df: pd.DataFrame) -> pd.DataFrame:
    fixed_headers = []
    current_channel = None

    for i, col in enumerate(df.columns):
        header_clean = str(col).strip()
        col_data = df.iloc[:, i]

        match = re.match(r"Channel\s+(\d+)", header_clean)
        if match:
            current_channel = match.group(1)
            fixed_headers.append(header_clean)
        elif (not header_clean or header_clean.startswith("Unnamed")) and col_data.dropna().empty:
            # Column is fully empty — drop it later
            fixed_headers.append(None)
        elif (not header_clean or header_clean.startswith("Unnamed")) and current_channel:
            # Header is unnamed but data is present — assign temp label
            fixed_headers.append(f"Temperature {current_channel}")
        else:
            fixed_headers.append(header_clean)

    # Apply cleaned headers
    df.columns = fixed_headers

    # Drop columns that are fully empty and were marked as None
    df = df.loc[:, df.columns.notna()]

    return df
