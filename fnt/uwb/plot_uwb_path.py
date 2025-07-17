## define and load the plot_uwb_path function
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon

def plot_uwb_path(df, filter_by="all", filter_day="all", arena_coordinates="none"):
    """
    Plots the smoothed trajectories of animals based on the specified filter.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the tracking data.
    filter_by (str): Can be "males", "females", "[specific ID]", or "all".
    filter_day (str or int): Defines the day(s) to include in the plot. Can be "all", a single day (e.g., 1), or a range (e.g., "1,5").
    arena_coordinates (str or pd.DataFrame): DataFrame with arena coordinates or "none". Default is "none".
    """
    # Filter the DataFrame based on the filter_by argument
    if filter_by.lower() == "males":
        df_filtered = df[df['sex'] == 'M']
    elif filter_by.lower() == "females":
        df_filtered = df[df['sex'] == 'F']
    elif filter_by.lower() != "all":
        df_filtered = df[df['ID'] == filter_by]
    else:
        df_filtered = df  # No filtering, use the entire DataFrame

    # Filter the DataFrame based on the filter_day argument
    if filter_day != "all":
        # Parse the filter_day argument
        day_range = [int(day) for day in str(filter_day).split(',')]
        if len(day_range) == 1:
            # Filter for a single day
            df_filtered = df_filtered[df_filtered['Day'] == day_range[0]]
        elif len(day_range) == 2:
            # Filter for a range of days
            df_filtered = df_filtered[(df_filtered['Day'] >= day_range[0]) & (df_filtered['Day'] <= day_range[1])]

    # Plot trajectory across days for the filtered data
    sns.set(style="whitegrid")
    g = sns.FacetGrid(df_filtered, col="Day", col_wrap=4, height=3, aspect=1.5, hue="ID", palette="tab10")

    # Plot each animal's trajectory with distinct colors
    g.map(plt.plot, "smoothed_x", "smoothed_y", marker="o", linestyle='-', linewidth=0.5, markersize=0.75, zorder=1)

    # Set titles and labels
    g.set_axis_labels("X Coordinate (meters)", "Y Coordinate (meters)")
    g.set_titles("Day {col_name}")

    # Check if arena_coordinates is provided
    if isinstance(arena_coordinates, pd.DataFrame):
        # Validate the columns of the DataFrame
        expected_columns = ['zone', 'x', 'y']
        if list(arena_coordinates.columns) != expected_columns:
            print("arena_coordinates argument expects a dataframe with three columns: zone, x, y.")
            return

        # Draw polygons for each zone
        for ax in g.axes.flat:
            for zone in arena_coordinates['zone'].unique():
                zone_coords = arena_coordinates[arena_coordinates['zone'] == zone]
                polygon = Polygon(zone_coords[['x', 'y']].values, closed=True, edgecolor='black', linewidth=2, facecolor='none', zorder=2)
                ax.add_patch(polygon)
                # Calculate the center of the polygon to place the label
                center_x = zone_coords['x'].mean()
                center_y = zone_coords['y'].mean()
                ax.text(center_x, center_y, zone, ha='center', va='center', fontsize=8, color='black', zorder=3)

    else:
        print("Plot produced with no arena coordinates.")

    # Adjust legend to show only distinct animal codes
    g.add_legend(title="Animal Code")
    plt.show()

# Example usage
# plot_path(df, filter_by="all", filter_day="all", arena_coordinates=ArenaCoords)