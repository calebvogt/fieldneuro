#!/usr/bin/env python3
"""
Minimal test for tag metadata GUI functionality
"""

import tkinter as tk
from tkinter import messagebox

def test_tag_metadata_gui():
    """Test just the tag metadata collection part"""
    
    # Simulate some tag IDs
    unique_tag_ids = [1, 2, 3, 4, 5]
    
    tag_window = tk.Tk()
    tag_window.title("Tag Metadata Test")
    tag_window.geometry("500x400")
    
    # Create scrollable frame
    canvas = tk.Canvas(tag_window)
    scrollbar = tk.Scrollbar(tag_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Header
    tk.Label(scrollable_frame, text="Tag Selection and Metadata", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
    tk.Label(scrollable_frame, text="(Leave sex/name blank for gray color)", font=("Arial", 9, "italic")).grid(row=1, column=0, columnspan=4, pady=(0,5))
    tk.Label(scrollable_frame, text="Include", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Tag ID", font=("Arial", 10, "bold")).grid(row=2, column=1, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Sex (M/F)", font=("Arial", 10, "bold")).grid(row=2, column=2, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Display ID", font=("Arial", 10, "bold")).grid(row=2, column=3, padx=5, pady=5)
    
    # Store tag metadata
    tag_metadata = {}
    
    for i, tag_id in enumerate(unique_tag_ids):
        row = i + 3
        
        # Include checkbox
        include_var = tk.BooleanVar()
        include_var.set(True)
        include_check = tk.Checkbutton(scrollable_frame, variable=include_var)
        include_check.grid(row=row, column=0, padx=5, pady=2)
        
        # Tag ID label
        tk.Label(scrollable_frame, text=str(tag_id)).grid(row=row, column=1, padx=5, pady=2)
        
        # Sex entry
        sex_var = tk.StringVar()
        sex_entry = tk.Entry(scrollable_frame, textvariable=sex_var, width=8)
        sex_entry.grid(row=row, column=2, padx=5, pady=2)
        
        # Display ID entry
        display_var = tk.StringVar()
        display_entry = tk.Entry(scrollable_frame, textvariable=display_var, width=15)
        display_entry.grid(row=row, column=3, padx=5, pady=2)
        
        tag_metadata[tag_id] = {
            'include': include_var,
            'sex': sex_var,
            'display_id': display_var
        }
    
    # Store results
    tag_results = {}
    
    def on_tag_submit():
        print("Collecting tag metadata from test GUI...")
        for tag_id, vars in tag_metadata.items():
            include = vars['include'].get()
            sex = vars['sex'].get().upper().strip() if vars['sex'].get() else ""
            display_id = vars['display_id'].get().strip() if vars['display_id'].get() else ""
            
            # Validate sex input
            if sex and sex not in ['M', 'F']:
                sex = ""
            
            tag_results[tag_id] = {
                'include': include,
                'sex': sex,
                'display_id': display_id
            }
            print(f"  Tag {tag_id}: include={include}, sex='{sex}', display_id='{display_id}'")
        
        tag_window.quit()
        tag_window.destroy()
        
        # Show results
        result_text = "Tag Metadata Results:\n\n"
        for tag_id, data in tag_results.items():
            result_text += f"Tag {tag_id}: include={data['include']}, sex='{data['sex']}', display_id='{data['display_id']}'\n"
        
        messagebox.showinfo("Results", result_text)
    
    tk.Button(scrollable_frame, text="Test Submit", command=on_tag_submit).grid(row=len(unique_tag_ids)+4, column=0, columnspan=4, pady=10)
    
    # Pack the scrollable components
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    tag_window.mainloop()

if __name__ == "__main__":
    test_tag_metadata_gui()
