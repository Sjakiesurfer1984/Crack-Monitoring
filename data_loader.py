import pandas as pd
import logging
from typing import Union, IO
import csv
from io import StringIO
import chardet
import re
import io
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
        Computes calibrated row-to-row differences for all 'Channel X' columns found in self.headers.
        Adds new columns like 'Channel X crack profile' to the end of the DataFrame.
        """
        # print(f"The entire DataFrame:\n{df}")
        self.headers = df.columns.tolist()  # Store headers for later use
        # print(f"Headers found in DataFrame:\n{self.headers}")
        # Dynamically find channel columns
        channel_cols = [col for col in self.headers if col.startswith("Channel")]
        # print(f"Channel columns found:\n{channel_cols}")
        if len(channel_cols) != len(calibration_factors):
            raise ValueError(f"Expected {len(channel_cols)} calibration factors, got {len(calibration_factors)}.")

        diff_columns = {}

        # for every channel in channel columns
        for i, channel in enumerate(channel_cols):
            if channel not in df.columns:
                logger.warning(f"⚠️ {channel} not found in DataFrame. Skipping.")
                continue
            # create a new diff column
            diff_col = f"{channel} crack profile"
            # logger.info(f"Diff. column: {diff_col}")
            # print(f"Df channel: {df[channel]} and df[channel].iloc[0]: {df[channel].iloc[0]}    and calibration factor: {calibration_factors[i]}")
            diff_values = (df[channel] - df[channel].iloc[0]) * calibration_factors[i]

            # logger.info(f"Diff. values: {diff_values}")
            diff_columns[diff_col] = diff_values.round(6)
            # logger.info(f"Final col result {diff_columns[diff_col]}")

        # Append new diff columns to the end of the DataFrame (axis=1 means we append them as columns)
        df = pd.concat([df, pd.DataFrame(diff_columns)], axis=1)

        return df

    def load_and_clean(self) -> pd.DataFrame:
        # Inform the logs that loading is starting
        logger.info("\U0001F504 Loading data...")

        try:
            # -------------------------------
            # Handle Excel file types (.xls/.xlsx)
            # -------------------------------
            if self.file_source.name.endswith((".xls", ".xlsx")):
                # Uses pandas to read an Excel file, skipping the first 23 metadata rows
                # header=0 means it now treats the 24th row (index 23) as the header row, after skipping the first 23 rows.
                df = pd.read_excel(self.file_source, sheet_name=0, skiprows=23, header=0)
                # we select the second row (index 1) and beyond, because the first row (index 0) is an empty row that we don't want.
                # we reset the index to remove the old index and create a new one, such that the first row becomes index 0 again. 
                df = df.iloc[1:].reset_index(drop=True)
                print(f"\nThe loaded dataframe:\n{df}")

            # -------------------------------
            # Handle CSV file type
            # -------------------------------
            elif self.file_source.name.endswith(".csv"):
                logger.info("🔄 Starting CSV load and clean")

                # 1) rewind + read raw bytes → text buffer
                self.file_source.seek(0)
                raw = self.file_source.read()
                text = raw.decode("utf-8", errors="ignore")  # force utf-8, drop bad chars
                buf = io.StringIO(text)

                # 2) read CSV, drop first 23 rows, no index_col
                df = pd.read_csv(
                    buf,
                    sep=",",
                    skiprows=23,       # drop lines 1–23
                    header=0,          # line 24 becomes header
                    index_col=False,   # do not treat any column as index
                    engine="python"    # more forgiving parser
                )
                print("\n[1] After read_csv (skiprows=23):")
                print(df.head(5))

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

                logger.info(f"✅ Finished cleaning. Final shape: {df.shape}")
                return df
 
  
            # -------------------------------
            # Unsupported file extension
            # -------------------------------
            else:
                raise ValueError("Unsupported file type. Please upload .csv, .xls, or .xlsx")

            # -------------------------------
            # Final log and return result
            # -------------------------------
            logger.info(f"\U0001F4CA Loaded file with shape: {df.shape}")
            return df

        except Exception as e:
            # Log error and raise it up to be handled by caller
            logger.error(f"\u274C Failed to load file: {e}")
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
