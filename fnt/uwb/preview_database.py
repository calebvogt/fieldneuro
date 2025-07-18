import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import ttk
import threading

def preview_database(file_path):
    def run_preview():
        # Connect to SQLite database
        conn = sqlite3.connect(file_path)

        # Fetch table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            conn.close()
            return

        # Use the first table for preview
        table_name = tables[0][0]
        print(f"Previewing table: {table_name}")

        # Load the first few rows of the table into a DataFrame
        query = f"SELECT * FROM {table_name} LIMIT 20;"
        df = pd.read_sql_query(query, conn)

        # Create a Tkinter window
        root = tk.Tk()
        root.title(f"Preview of {table_name}")

        # Create a Treeview widget
        tree = ttk.Treeview(root, columns=list(df.columns), show="headings")

        # Define column headings
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # Insert data into the Treeview
        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        # Add a horizontal scrollbar
        h_scroll = ttk.Scrollbar(root, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(side="bottom", fill="x")

        # Add a vertical scrollbar
        v_scroll = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=v_scroll.set)
        v_scroll.pack(side="right", fill="y")

        tree.pack(expand=True, fill="both")

        root.mainloop()
        conn.close()

    # Run the preview in a separate thread
    threading.Thread(target=run_preview, daemon=True).start()
