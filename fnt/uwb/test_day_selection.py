import tkinter as tk
from datetime import date

def test_day_selection():
    """Test the day selection logic"""
    print("Testing day selection window...")
    
    # Sample dates
    test_dates = [
        date(2025, 3, 27),
        date(2025, 3, 28),
        date(2025, 3, 29),
        date(2025, 3, 30)
    ]
    
    day_window = tk.Tk()
    day_window.title("Test Day Selection")
    
    selected_days = {}
    
    tk.Label(day_window, text="Select test days:").pack(pady=10)
    
    for test_date in test_dates:
        var = tk.BooleanVar()
        var.set(True)  # Default checked
        selected_days[test_date] = var
        tk.Checkbutton(day_window, text=str(test_date), variable=var).pack(anchor="w")
    
    def on_submit():
        day_window.quit()
        day_window.destroy()
    
    tk.Button(day_window, text="Continue", command=on_submit).pack(pady=10)
    
    print("Showing window...")
    day_window.mainloop()
    
    # Collect results after window closes
    selected_date_list = [date for date, var in selected_days.items() if var.get()]
    print(f"Selected dates: {selected_date_list}")
    print(f"Number of selected dates: {len(selected_date_list)}")
    
    # Verify the variables still exist and have values
    for test_date, var in selected_days.items():
        print(f"Date {test_date}: {var.get()}")

if __name__ == "__main__":
    test_day_selection()
