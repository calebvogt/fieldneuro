import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import sqlite3
from mpl_toolkits.mplot3d import Axes3D

def create_daily_faceted_plots(data, tag_color_map, tag_label_map, arena_coordinates=None):
    """
    Create daily faceted plots showing trajectories across days using seaborn FacetGrid style.
    Based on the plot_uwb_path function from the FieldNeuroToolbox.
    """
    print("Creating daily faceted plots...")
    
    # Set column names for plotting
    x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
    
    # Create a day number for faceting (assuming chronological order)
    data = data.copy()
    unique_dates = sorted(data['Date'].unique())
    date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
    data['Day'] = data['Date'].map(date_to_day)
    
    # Add display labels to data for plotting
    data['Display_Label'] = data['shortid'].map(tag_label_map)
    
    try:
        # Try to use seaborn FacetGrid for professional-looking plots
        print("  Using seaborn FacetGrid for daily faceted plots...")
        
        # Set seaborn style
        sns.set(style="whitegrid")
        
        # Create a single faceted plot with all animals
        print(f"  Creating combined daily faceted plot with all {len(data['shortid'].unique())} animals...")
        
        # Create custom color palette based on our tag_color_map
        unique_tags = sorted(data['shortid'].unique())
        palette = [tag_color_map[tag] for tag in unique_tags]
        
        # Create FacetGrid
        g = sns.FacetGrid(data, col="Day", col_wrap=4, height=3, aspect=1.2, 
                         hue="Display_Label", palette=palette, margin_titles=True)
        
        # Plot trajectories with markers
        g.map(plt.plot, x_col, y_col, marker="o", linestyle='-', 
              linewidth=1.5, markersize=0.75, alpha=0.8, zorder=1)
        
        # Set axis labels and titles
        g.set_axis_labels("X Coordinate (meters)", "Y Coordinate (meters)")
        g.set_titles("Day {col_name}")
        
        # Add arena boundaries if provided
        if isinstance(arena_coordinates, pd.DataFrame):
            print("  Adding arena zone boundaries...")
            expected_columns = ['zone', 'x', 'y']
            if list(arena_coordinates.columns) == expected_columns:
                # Draw polygons for each zone on all subplots
                for ax in g.axes.flat:
                    for zone in arena_coordinates['zone'].unique():
                        zone_coords = arena_coordinates[arena_coordinates['zone'] == zone]
                        polygon = patches.Polygon(zone_coords[['x', 'y']].values, closed=True, 
                                               edgecolor='black', linewidth=2, facecolor='none', zorder=2)
                        ax.add_patch(polygon)
                        # Calculate center of polygon for label
                        center_x = zone_coords['x'].mean()
                        center_y = zone_coords['y'].mean()
                        ax.text(center_x, center_y, zone, ha='center', va='center', 
                               fontsize=8, color='black', fontweight='bold', zorder=3)
            else:
                print("    Warning: Arena coordinates DataFrame has incorrect columns. Expected: zone, x, y")
        
        # Add legend
        g.add_legend(title="Animal", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and show
        plt.subplots_adjust(right=0.85)  # Make room for legend
        plt.show(block=False)
        print("  Combined daily faceted plot complete.")
        
        # Also create individual plots for each animal for detailed viewing
        print("  Creating individual daily faceted plots for each animal...")
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            
            if tag_data.empty:
                continue
                
            color = tag_color_map[tag]
            label = tag_label_map[tag]
            
            print(f"    Creating individual daily plot for {label}")
            
            # Create individual FacetGrid for this animal
            g_individual = sns.FacetGrid(tag_data, col="Day", col_wrap=4, height=3, aspect=1.2, 
                                       margin_titles=True)
            
            # Plot with animal's specific color
            g_individual.map(plt.plot, x_col, y_col, color=color, marker="o", linestyle='-', 
                           linewidth=2, markersize=1, alpha=0.9, zorder=1)
            
            # Set labels and title
            g_individual.set_axis_labels("X Coordinate (meters)", "Y Coordinate (meters)")
            g_individual.set_titles("Day {col_name}")
            g_individual.fig.suptitle(f'Daily Trajectories - {label}', fontsize=14, fontweight='bold', y=1.02)
            
            # Add arena boundaries if provided
            if isinstance(arena_coordinates, pd.DataFrame):
                expected_columns = ['zone', 'x', 'y']
                if list(arena_coordinates.columns) == expected_columns:
                    for ax in g_individual.axes.flat:
                        for zone in arena_coordinates['zone'].unique():
                            zone_coords = arena_coordinates[arena_coordinates['zone'] == zone]
                            polygon = patches.Polygon(zone_coords[['x', 'y']].values, closed=True, 
                                                   edgecolor='black', linewidth=2, facecolor='none', zorder=2)
                            ax.add_patch(polygon)
                            center_x = zone_coords['x'].mean()
                            center_y = zone_coords['y'].mean()
                            ax.text(center_x, center_y, zone, ha='center', va='center', 
                                   fontsize=8, color='black', fontweight='bold', zorder=3)
            
            plt.tight_layout()
            plt.show(block=False)
        
    except Exception as e:
        print(f"  Seaborn plotting failed ({e}), falling back to matplotlib subplots...")
        
        # Fallback to matplotlib subplots if seaborn fails
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            
            if tag_data.empty:
                continue
                
            color = tag_color_map[tag]
            label = tag_label_map[tag]
            
            print(f"    Creating matplotlib daily plot for {label}")
            
            unique_days = sorted(tag_data['Day'].unique())
            n_days = len(unique_days)
            
            # Calculate subplot grid (4 columns max)
            cols = min(4, n_days)
            rows = (n_days + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            fig.suptitle(f'Daily Trajectories - {label}', fontsize=16, fontweight='bold')
            
            # Handle different subplot configurations
            if n_days == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()
            
            # Calculate consistent axis limits
            x_min, x_max = tag_data[x_col].min(), tag_data[x_col].max()
            y_min, y_max = tag_data[y_col].min(), tag_data[y_col].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = 0.1 * x_range if x_range > 0 else 0.5
            y_padding = 0.1 * y_range if y_range > 0 else 0.5
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding
            
            for i, day in enumerate(unique_days):
                ax = axes[i]
                day_data = tag_data[tag_data['Day'] == day]
                
                if not day_data.empty:
                    ax.plot(day_data[x_col], day_data[y_col], 
                           color=color, linewidth=1.5, alpha=0.8, marker='o', markersize=0.5)
                
                ax.set_xlabel('X Coordinate (meters)')
                ax.set_ylabel('Y Coordinate (meters)')
                ax.set_title(f'Day {day}')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            
            # Hide unused subplots
            for i in range(n_days, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show(block=False)
    
    print("  Daily faceted plots complete.")

def create_occupancy_heatmap_2d(data, tag_color_map, tag_label_map):
    """
    Create 2D occupancy heatmaps for each tag showing spatial usage across days.
    Based on plot_occupancy_heatmap_2d from the notebook.
    """
    print("Creating 2D occupancy heatmaps...")
    
    # Set column names for plotting
    x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
    
    # Ensure we have Day column
    if 'Day' not in data.columns:
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
    
    # Create heatmaps for each tag
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag]
        
        if tag_data.empty:
            continue
            
        label = tag_label_map[tag]
        print(f"  Creating 2D occupancy heatmap for {label}")
        
        # Get unique days for this animal
        unique_days = sorted(tag_data['Day'].unique())
        
        # Create a list to store the data for each day
        data_list = []
        
        for day in unique_days:
            # Filter the DataFrame for the specific day
            day_data = tag_data[tag_data['Day'] == day]
            
            if day_data.empty:
                continue
                
            # Create a 2D histogram of the x and y coordinates
            x = day_data[x_col]
            y = day_data[y_col]
            occupancy, xedges, yedges = np.histogram2d(x, y, bins=50)
            
            # Store the data in a dictionary
            data_list.append({
                'day': day,
                'occupancy': occupancy.T
            })
        
        if not data_list:
            continue
            
        # Create a facet grid with 4 plots per row
        num_days = len(data_list)
        num_cols = 4
        num_rows = (num_days + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))
        fig.suptitle(f'2D Occupancy Heatmaps - {label}', fontsize=16, fontweight='bold')
        
        # Flatten the axes array for easy iteration
        if num_days == 1:
            axes = [axes]
        elif num_rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, day_data in enumerate(data_list):
            ax = axes[i]
            sns.heatmap(day_data['occupancy'], ax=ax, cmap='viridis', cbar=True, 
                       xticklabels=False, yticklabels=False)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f'Day {day_data["day"]}')
        
        # Hide any unused subplots
        for i in range(len(data_list), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show(block=False)
    
    print("  2D occupancy heatmaps complete.")

def create_occupancy_heatmap_3d(data, tag_color_map, tag_label_map, occupancy_scale="daily"):
    """
    Create 3D occupancy heatmaps for each tag showing spatial usage across days.
    Based on plot_occupancy_heatmap from the notebook.
    """
    print("Creating 3D occupancy heatmaps...")
    
    # Set column names for plotting
    x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
    
    # Ensure we have Day column
    if 'Day' not in data.columns:
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
    
    # Create 3D heatmaps for each tag
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag]
        
        if tag_data.empty:
            continue
            
        label = tag_label_map[tag]
        print(f"  Creating 3D occupancy heatmap for {label}")
        
        # Get unique days for this animal
        unique_days = sorted(tag_data['Day'].unique())
        
        # Create a list to store the data for each day
        data_list = []
        max_occupancy = 0
        
        for day in unique_days:
            # Filter the DataFrame for the specific day
            day_data = tag_data[tag_data['Day'] == day]
            
            if day_data.empty:
                continue
                
            # Create a 2D histogram of the x and y coordinates
            x = day_data[x_col]
            y = day_data[y_col]
            occupancy, xedges, yedges = np.histogram2d(x, y, bins=50)
            
            # Update the maximum occupancy if needed
            if occupancy_scale == "all":
                max_occupancy = max(max_occupancy, occupancy.max())
            
            # Create meshgrid for the surface plot
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(xcenters, ycenters)
            
            # Store the data in a dictionary
            data_list.append({
                'day': day,
                'X': X,
                'Y': Y,
                'occupancy': occupancy.T
            })
        
        if not data_list:
            continue
            
        # Create a facet grid with 4 plots per row
        num_days = len(data_list)
        num_cols = 4
        num_rows = (num_days + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows), 
                               subplot_kw={'projection': '3d'})
        fig.suptitle(f'3D Occupancy Heatmaps - {label}', fontsize=16, fontweight='bold')
        
        # Flatten the axes array for easy iteration
        if num_days == 1:
            axes = [axes]
        elif num_rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, day_data in enumerate(data_list):
            ax = axes[i]
            ax.plot_surface(day_data['X'], day_data['Y'], day_data['occupancy'], 
                          cmap='viridis', edgecolor='none')
            ax.set_xlabel('X Coordinate (m)')
            ax.set_ylabel('Y Coordinate (m)')
            ax.set_zlabel('Occupancy')
            ax.set_title(f'Day {day_data["day"]}')
            
            # Set the viewing angle
            ax.view_init(elev=30, azim=250)
            
            # Set the z-axis limit based on occupancy_scale
            if occupancy_scale == "all":
                ax.set_zlim(0, max_occupancy)
        
        # Hide any unused subplots
        for i in range(len(data_list), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show(block=False)
    
    print("  3D occupancy heatmaps complete.")

def create_actograms(data, tag_color_map, tag_label_map, velocity_threshold=0.1):
    """
    Create actogram plots for each tag showing circadian activity patterns.
    Based on plot_actogram from the notebook.
    """
    print("Creating actograms...")
    
    # Set column names for plotting
    x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
    
    # Ensure we have Day column
    if 'Day' not in data.columns:
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
    
    # Create actograms for each tag
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag].copy()
        
        if tag_data.empty:
            continue
            
        label = tag_label_map[tag]
        print(f"  Creating actogram for {label}")
        
        # Sort by timestamp and calculate velocity
        tag_data = tag_data.sort_values(by='Timestamp')
        
        # Calculate velocity using proper groupby to avoid mixing data between sessions
        tag_data['velocity'] = np.sqrt(
            tag_data[x_col].diff()**2 + tag_data[y_col].diff()**2
        ) / tag_data['Timestamp'].diff().dt.total_seconds()
        
        # Define activity as velocity above the specified threshold
        tag_data['active'] = tag_data['velocity'] > velocity_threshold
        
        # Create the actogram plot
        plt.figure(figsize=(8, 6))
        
        # Plot activity lines
        for day in tag_data['Day'].unique():
            day_data = tag_data[tag_data['Day'] == day]
            active_times = day_data[day_data['active']]['Timestamp']
            
            if not active_times.empty:
                hours = active_times.dt.hour + active_times.dt.minute / 60.0
                for hour in hours:
                    plt.plot([hour, hour], [day - 0.4, day + 0.4], color='black', linewidth=0.5)
        
        # Add gray shading for dark periods (assuming lights off at 7PM, on at 7AM)
        plt.axvspan(19, 24, color='gray', alpha=0.3, label='Dark period')
        plt.axvspan(0, 7, color='gray', alpha=0.3)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Trial')
        plt.title(f'Actogram - {label}')
        plt.gca().invert_yaxis()
        
        # Set y-axis ticks to show all days
        days = sorted(tag_data['Day'].unique())
        plt.yticks(days)
        
        # Set x-axis to show 24 hours
        plt.xlim(0, 24)
        plt.xticks(range(0, 25, 4))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
    
    print("  Actograms complete.")

def uwb_create_plots():
    """
    Standalone function to create multiple UWB tag plots with optional downsampling and smoothing.
    Creates multiple plots that display simultaneously for quick data exploration.
    """
    print("Starting UWB Tag Plot Generation...")

    # ===== COLLECT ALL USER CHOICES UPFRONT =====
    print("Collecting all user preferences...")
    
    # 1. Prompt user to select SQLite file
    root = tk.Tk()
    root.withdraw()
    print("Prompting user to select SQLite file...")
    file_path = filedialog.askopenfilename(title="Select SQLite File", filetypes=[("SQLite Files", "*.sqlite")])

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {file_path}")

    # 2. Ask user if they want to downsample to 1Hz
    print("Prompting user for downsampling option...")
    downsample = messagebox.askyesno("Downsample Data", "Do you want to downsample the data to 1Hz?")
    print(f"Downsampling option selected: {'Yes' if downsample else 'No'}")

    # 3. Create a new window for smoothing method selection
    print("Prompting user for smoothing method...")
    smoothing_window = tk.Tk()
    smoothing_window.title("Select Smoothing Method")

    smoothing_choice = tk.StringVar(value="none")

    def set_smoothing(choice):
        print(f"Button clicked: {choice}")
        smoothing_choice.set(choice)
        print("Attempting to quit and destroy smoothing window...")
        try:
            smoothing_window.quit()  # Exit the mainloop
            smoothing_window.destroy()  # Destroy the window
            print("Smoothing window successfully quit and destroyed.")
        except Exception as e:
            print(f"Error quitting/destroying smoothing window: {e}")

    tk.Label(smoothing_window, text="Choose a smoothing method:").pack(pady=10)
    tk.Button(smoothing_window, text="Savitzky-Golay", command=lambda: set_smoothing("savitzky-golay")).pack(pady=5)
    tk.Button(smoothing_window, text="Rolling-Average", command=lambda: set_smoothing("rolling-average")).pack(pady=5)
    tk.Button(smoothing_window, text="None", command=lambda: set_smoothing("none")).pack(pady=5)

    print("Starting smoothing window mainloop...")
    smoothing_window.mainloop()
    print("Exited smoothing window mainloop.")

    smoothing = smoothing_choice.get()
    print(f"Smoothing method selected: {smoothing}")

    # 4. Timezone selection
    print("Asking user about timezone...")
    timezone_window = tk.Tk()
    timezone_window.title("Select US Timezone")
    
    timezone_choice = tk.StringVar(value="UTC")
    
    def set_timezone(tz):
        timezone_choice.set(tz)
        timezone_window.quit()
        timezone_window.destroy()
    
    tk.Label(timezone_window, text="Select your timezone for timestamp conversion:").pack(pady=10)
    tk.Button(timezone_window, text="Eastern (ET)", command=lambda: set_timezone("US/Eastern")).pack(pady=3)
    tk.Button(timezone_window, text="Central (CT)", command=lambda: set_timezone("US/Central")).pack(pady=3)
    tk.Button(timezone_window, text="Mountain (MT)", command=lambda: set_timezone("US/Mountain")).pack(pady=3)
    tk.Button(timezone_window, text="Pacific (PT)", command=lambda: set_timezone("US/Pacific")).pack(pady=3)
    tk.Button(timezone_window, text="Alaska (AKT)", command=lambda: set_timezone("US/Alaska")).pack(pady=3)
    tk.Button(timezone_window, text="Hawaii (HST)", command=lambda: set_timezone("US/Hawaii")).pack(pady=3)
    tk.Button(timezone_window, text="Keep UTC", command=lambda: set_timezone("UTC")).pack(pady=3)
    
    timezone_window.mainloop()
    
    selected_timezone = timezone_choice.get()
    print(f"Selected timezone: {selected_timezone}")

    print("All user preferences collected! Now checking available tags...")
    print("=" * 50)

    # Quick database query to get available tags without loading all data
    print("Connecting to database to check available tags...")
    conn = sqlite3.connect(file_path)
    
    # Query unique tags
    tag_query = "SELECT DISTINCT shortid FROM data ORDER BY shortid"
    available_tags = pd.read_sql_query(tag_query, conn)
    conn.close()
    
    if available_tags.empty:
        print("No data found in database. Exiting.")
        return
        
    print(f"Found {len(available_tags)} unique tags in database: {list(available_tags['shortid'])}")

    # 5. Tag selection and metadata window
    print("Opening tag selection and metadata window...")
    unique_tag_ids = list(available_tags['shortid'])
    
    tag_window = tk.Tk()
    tag_window.title("Tag Selection and Metadata")
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
    tk.Label(scrollable_frame, text="(Optional: Enter sex M/F and display name for custom colors/labels)", font=("Arial", 9, "italic")).grid(row=1, column=0, columnspan=4, pady=(0,5))
    tk.Label(scrollable_frame, text="Include", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Tag ID", font=("Arial", 10, "bold")).grid(row=2, column=1, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Sex (M/F)", font=("Arial", 10, "bold")).grid(row=2, column=2, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Display ID", font=("Arial", 10, "bold")).grid(row=2, column=3, padx=5, pady=5)
    
    # Store tag metadata
    tag_metadata = {}
    tag_widgets = {}  # Store direct references to Entry widgets
    
    print(f"Creating GUI elements for {len(unique_tag_ids)} tags...")
    for i, tag_id in enumerate(unique_tag_ids):
        row = i + 3  # Changed from i + 2 to i + 3 to account for the new instruction row
        
        # Include checkbox
        include_var = tk.BooleanVar()
        include_var.set(True)  # Default to include all tags
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
        
        # Store direct widget references as backup
        tag_widgets[tag_id] = {
            'include': include_check,
            'sex_entry': sex_entry,
            'display_entry': display_entry
        }
        
        print(f"  Created GUI for tag {tag_id} at row {row}")
    
    print("GUI creation complete. Waiting for user input...")
    
    # Store results
    tag_results = {}
    
    def on_tag_submit():
        print("Collecting tag metadata from user input...")
        print("Method 1 - Raw metadata check from StringVar:")
        
        # First, let's check what's actually in the StringVar objects
        for tag_id, vars in tag_metadata.items():
            sex_raw = vars['sex'].get()
            display_raw = vars['display_id'].get()
            print(f"  Tag {tag_id} RAW: sex='{sex_raw}' (len={len(sex_raw)}), display='{display_raw}' (len={len(display_raw)})")
        
        print("Method 2 - Direct from Entry widgets:")
        # Try getting values directly from Entry widgets
        for tag_id, widgets in tag_widgets.items():
            sex_direct = widgets['sex_entry'].get()
            display_direct = widgets['display_entry'].get()
            print(f"  Tag {tag_id} DIRECT: sex='{sex_direct}' (len={len(sex_direct)}), display='{display_direct}' (len={len(display_direct)})")
        
        print("Processing metadata using direct method:")
        for tag_id, widgets in tag_widgets.items():
            include = tag_metadata[tag_id]['include'].get()
            sex_raw = widgets['sex_entry'].get()
            display_raw = widgets['display_entry'].get()
            
            # More careful string processing
            sex = sex_raw.strip().upper() if sex_raw else ""
            display_id = display_raw.strip() if display_raw else ""
            
            print(f"  Tag {tag_id} PROCESSED: sex_raw='{sex_raw}' -> sex='{sex}', display_raw='{display_raw}' -> display_id='{display_id}'")
            
            # Validate sex input
            if sex and sex not in ['M', 'F']:
                print(f"    WARNING: Invalid sex '{sex}' for tag {tag_id}, clearing...")
                sex = ""
            
            tag_results[tag_id] = {
                'include': include,
                'sex': sex,
                'display_id': display_id
            }
            print(f"  Tag {tag_id} FINAL: include={include}, sex='{sex}', display_id='{display_id}'")
        
        tag_window.quit()
        tag_window.destroy()
        print("Tag metadata collection complete.")
    
    tk.Button(scrollable_frame, text="Continue", command=on_tag_submit).grid(row=len(unique_tag_ids)+4, column=0, columnspan=4, pady=10)
    
    # Pack the scrollable components
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    tag_window.mainloop()
    
    # Process tag results
    selected_tags = [tag_id for tag_id, data in tag_results.items() if data['include']]
    print(f"Selected tags: {selected_tags}")
    
    if not selected_tags:
        print("No tags selected. Exiting.")
        return

    print("All choices finalized! Beginning data processing...")
    print("=" * 50)

    # ===== DATABASE OPERATIONS AND DATA PROCESSING =====
    
    # Connect to SQLite database and query data
    print("Connecting to the SQLite database...")
    conn = sqlite3.connect(file_path)
    print("Querying data from the database...")
    query = "SELECT * FROM data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    print("Data successfully imported.")

    print("Processing data...")
    data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
    
    # Convert timezone if not UTC
    if selected_timezone != "UTC":
        print(f"Converting timestamps to {selected_timezone}...")
        data['Timestamp'] = data['Timestamp'].dt.tz_convert(selected_timezone)
    
    data['location_x'] *= 0.0254
    data['location_y'] *= 0.0254
    data = data.sort_values(by=['shortid', 'Timestamp'])
    
    # Filter data to only selected tags
    print(f"Filtering data to selected tags: {selected_tags}")
    data = data[data['shortid'].isin(selected_tags)]
    print(f"Filtered data contains {len(data)} records")
    print("Data sorted by shortid and timestamp.")

    if downsample:
        print("Downsampling data to 1Hz...")
        data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
        data = data.groupby(['shortid', 'time_sec']).first().reset_index()
        print("Data successfully downsampled.")

    if smoothing == 'savitzky-golay':
        print("Applying Savitzky-Golay smoothing...")
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            return savgol_filter(group, window_length=window_length, polyorder=polyorder)

        data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
        data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
        print("Savitzky-Golay smoothing applied.")

    elif smoothing == 'rolling-average':
        print("Applying rolling-average smoothing...")
        data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        print("Rolling-average smoothing applied.")

    print("Generating plots...")
    
    # Create color map and labels based on tag metadata
    print("Setting up tag colors and labels based on metadata...")
    unique_tags = data['shortid'].unique()
    tag_color_map = {}
    tag_label_map = {}
    
    # Default color palette for tags without metadata
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
    
    print("Applying tag metadata and colors...")
    for i, tag in enumerate(unique_tags):
        # Convert tag to int if it's not already, to match tag_results keys
        tag_key = int(tag) if isinstance(tag, (str, float)) else tag
        
        if tag_key in tag_results:
            metadata = tag_results[tag_key]
            sex = metadata['sex']
            display_id = metadata['display_id']
            
            # Set color based on sex if provided, otherwise use default color
            if sex == 'M':
                color = 'blue'
            elif sex == 'F':
                color = 'red'
            else:
                # Use default color scheme if no sex specified
                color = default_colors[i % len(default_colors)]
            
            tag_color_map[tag] = color
            
            # Create label
            if sex and display_id:
                label = f"{sex}-{display_id}"
            elif sex:
                label = f"{sex}-{tag}"
            elif display_id:
                label = display_id
            else:
                label = f"Tag {tag}"
            
            tag_label_map[tag] = label
            print(f"  Tag {tag}: color={color}, label='{label}'")
        else:
            # Fallback for tags not in results (shouldn't happen with current logic)
            color = default_colors[i % len(default_colors)]
            tag_color_map[tag] = color
            tag_label_map[tag] = f"Tag {tag}"
            print(f"  Tag {tag}: color={color}, label='Tag {tag}' (fallback)")

    # Add date column for daily faceted plots
    data['Date'] = data['Timestamp'].dt.date
    
    # Set matplotlib to non-blocking mode for simultaneous plots
    plt.ion()  # Turn on interactive mode
    
    # Increase the figure limit to handle multiple plots
    plt.rcParams['figure.max_open_warning'] = 50
    
    # Plot combined view with all tags
    print("Creating combined plot with all tags...")
    plt.figure(figsize=(10, 8))
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag]
        x_col = 'smoothed_x' if 'smoothed_x' in tag_data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in tag_data.columns else 'location_y'

        color = tag_color_map[tag]
        label = tag_label_map[tag]
        plt.plot(tag_data[x_col], tag_data[y_col], label=label, color=color)

    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('UWB Tag Paths - Combined View')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)  # Non-blocking show
    print("Combined plot complete.")
    
    # Plot individual views for each tag
    print("Creating individual plots for each tag...")
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag]
        x_col = 'smoothed_x' if 'smoothed_x' in tag_data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in tag_data.columns else 'location_y'

        color = tag_color_map[tag]
        label = tag_label_map[tag]
        
        plt.figure(figsize=(10, 8))
        plt.plot(tag_data[x_col], tag_data[y_col], label=label, color=color, linewidth=2)
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.title(f'UWB Tag Path - {label}')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)  # Non-blocking show
        print(f"Individual plot for {label} complete.")
    
    # Create daily faceted plots for each tag
    print("Creating daily faceted plots for each tag...")
    create_daily_faceted_plots(data, tag_color_map, tag_label_map, arena_coordinates=None)
    
    # Create 2D occupancy heatmaps for each tag
    print("Creating 2D occupancy heatmaps for each tag...")
    create_occupancy_heatmap_2d(data, tag_color_map, tag_label_map)
    
    # Create 3D occupancy heatmaps for each tag
    print("Creating 3D occupancy heatmaps for each tag...")
    create_occupancy_heatmap_3d(data, tag_color_map, tag_label_map, occupancy_scale="daily")
    
    # Create actograms for each tag
    print("Creating actograms for each tag...")
    create_actograms(data, tag_color_map, tag_label_map, velocity_threshold=0.1)
    
    # Keep plots open - user can interact with them
    print("\n" + "=" * 50)
    print("🎉 All plots generated successfully!")
    print("📊 All plots are now displayed and interactive.")
    print("💡 You can zoom, pan, and interact with each plot window.")
    print("🔧 Close individual plot windows when you're done with them.")
    print("⚠️  Note: Plots will remain open until you close them manually.")
    print("=" * 50)
    
    # Keep the script running so plots stay open
    input("Press Enter to close all plots and exit...")
    plt.close('all')
    print("All plotting complete.")

if __name__ == "__main__":
    uwb_create_plots()
