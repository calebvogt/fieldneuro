import pandas as pd
import sqlite3
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def uwb_convert_database():
    """
    Converts an SQLite UWB database file to a raw CSV export of the `data` table.
    Prompts user to select the `.sqlite` file via a file dialog.
    Saves CSV to the same directory as the input file.
    """
    # Hide the root Tkinter window
    Tk().withdraw()

    # Ask user for SQLite file
    print("Opening file dialog...")
    db_file = askopenfilename(title="Select UWB SQLite database file", filetypes=[("SQLite files", "*.sqlite")])
    if not db_file:
        print("No file selected.")
        return

    print(f"Selected: {db_file}")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)

    # Read the entire `data` table
    try:
        df = pd.read_sql_query("SELECT * FROM data", conn)
        print(f"Read {len(df)} rows from 'data' table.")
    except Exception as e:
        print(f"Error reading table: {e}")
        conn.close()
        return

    # Close DB connection
    conn.close()

    # Save to CSV in same folder as original file
    out_path = os.path.splitext(db_file)[0] + ".csv"
    df.to_csv(out_path, index=False)
    print(f"Saved CSV to: {out_path}")
