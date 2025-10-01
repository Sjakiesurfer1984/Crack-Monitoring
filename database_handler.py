import apsw
import sqlite3
import pandas as pd
import logging
import os
from typing import Optional, List

# Constants
# Place the SQLite file right next to this module
# DB_PATH = os.path.join(os.path.dirname(__file__), "sensor_data.db")
DB_PATH = r"C:\Users\TimVos\AIE\Projects Aus - General\2024\24-0397 PP Spodumene Shed Concrete Wall Crack Monitoring\5 Project Documents\3. Data\sensor_data.db"
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        logger.info(f"Database path set to: {self.db_path}")

    def _connect(self) -> apsw.Connection:
        """Create or open an APSW connection."""
        parent = os.path.dirname(self.db_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        conn = apsw.Connection(self.db_path)
        logger.info(f"🗄 Connected to APSW database at {self.db_path}")
        return conn

    def _table_exists(self, conn: apsw.Connection) -> bool:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_values';"
        try:
            exists = bool(conn.execute(query).fetchone())
            return exists
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False

    def save_to_db(self, df: pd.DataFrame) -> None:
        """
        Save DataFrame to the database, handling inserts and updates.
        Uses APSW for deletes and sqlite3 for pandas to_sql integration.
        """
        if df.empty:
            logger.warning("⚠️ Attempted to save empty DataFrame to DB.")
            return

        # Prepare DataFrame
        df = df.copy()
        if 'Date/time' in df.columns:
            df.rename(columns={'Date/time': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce').dt.floor('min')
        print(f"We now have our DF with datetime:\n{df.head(5)}")
        df.dropna(subset=['datetime'], inplace=True)
        print(f"We now have dropped the NaN's in the DateTime column:\n{df.head(5)}")
        # Check table existence via APSW
        apsw_conn = self._connect()
        table_exists = self._table_exists(apsw_conn)
        apsw_conn.close()

        # Use sqlite3 for pandas to_sql
        dbapi_conn = sqlite3.connect(self.db_path)
        try:
            if table_exists:
                logger.info("📥 Checking for existing rows in DB...")
                existing = pd.read_sql_query(
                    "SELECT datetime FROM sensor_values", dbapi_conn, parse_dates=['datetime']
                )
                existing['datetime'] = existing['datetime'].dt.floor('min')
                existing_set = set(existing['datetime'].dropna())

                # Identify overlaps and new rows
                df_overlap = df[df['datetime'].isin(existing_set)]
                df_new = df[~df['datetime'].isin(existing_set)]

                # Delete overlapping rows via APSW
                if not df_overlap.empty:
                    apsw_conn = self._connect()
                    placeholders = ','.join('?' for _ in df_overlap)
                    apsw_conn.execute(
                        f"DELETE FROM sensor_values WHERE datetime IN ({placeholders})",
                        tuple(df_overlap['datetime'])
                    )
                    apsw_conn.close()
                    logger.info(f"🧹 Deleted {len(df_overlap)} existing rows to update.")

                # Append new rows
                to_insert = df_new if not df_overlap.empty else df
                to_insert.to_sql('sensor_values', dbapi_conn, if_exists='append', index=False)
                logger.info(f"✅ Inserted {len(to_insert)} rows (new + updated)")
            else:
                # Create table from scratch
                df.to_sql('sensor_values', dbapi_conn, if_exists='replace', index=False)
                logger.info("📦 Created new table and inserted all data")
        except Exception as e:
            logger.error(f"❌ Failed during DB insert: {e}")
            raise
        finally:
            dbapi_conn.close()

    def query_data(self, start_date: str, end_date: str, selected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        with self._connect() as conn:
            if not self._table_exists(conn):
                logger.warning("⚠️ Table 'sensor_values' does not exist.")
                return pd.DataFrame()
            try:
                all_cols = pd.read_sql_query("PRAGMA table_info(sensor_values);", conn)["name"].tolist()
                if selected_columns:
                    valid_cols = ['datetime'] + [c for c in selected_columns if c in all_cols and c != 'datetime']
                else:
                    valid_cols = all_cols
                select_clause = ", ".join(f'"{c}"' for c in valid_cols)
                query = f"SELECT {select_clause} FROM sensor_values WHERE datetime BETWEEN ? AND ? ORDER BY datetime;"
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                return df
            except Exception as e:
                logger.error(f"❌ Failed to query data: {e}")
                return pd.DataFrame()

    def load_all_data(self) -> pd.DataFrame:
        with self._connect() as conn:
            if not self._table_exists(conn):
                logger.warning("📭 No data found in DB (table missing).")
                return pd.DataFrame()
            try:
                df = pd.read_sql_query("SELECT * FROM sensor_values", conn)
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                return df
            except Exception as e:
                logger.error(f"❌ Failed to load all data: {e}")
                return pd.DataFrame()

    def get_latest_datetime(self) -> Optional[pd.Timestamp]:
        with self._connect() as conn:
            if not self._table_exists(conn):
                logger.warning("⚠️ No 'sensor_values' table found.")
                return None
            try:
                result = pd.read_sql_query("SELECT MAX(datetime) as max_dt FROM sensor_values", conn)
                if result.empty or pd.isna(result['max_dt'].iloc[0]):
                    return None
                return pd.to_datetime(result['max_dt'].iloc[0])
            except Exception as e:
                logger.error(f"❌ Failed to retrieve latest datetime: {e}")
                return None
