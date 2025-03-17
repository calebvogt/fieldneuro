import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_path(df, filter_by="all", filter_day="all"):
    """
    Plots the smoothed trajectories of animals based on the specified filter.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the tracking data.
    filter_by (str): Can be "males", "females", "[specific code]", or "all".
    filter_day (str or int): Defines the day(s) to include in the plot. Can be "all", a single day (e.g., 1), or a range (e.g., "1,5").
    """
    # Filter the DataFrame based on the filter_by argument
    if filter_by.lower() == "males":
        df_filtered = df[df['sex'] == 'M']
    elif filter_by.lower() == "females":
        df_filtered = df[df['sex'] == 'F']
    elif filter_by.lower() != "all":
        df_filtered = df[df['code'] == filter_by]
    else:
        df_filtered = df  # No filtering, use the entire DataFrame

    # Filter the DataFrame based on the filter_day argument
    if filter_day != "all":
        # Parse the filter_day argument
        day_range = [int(day) for day in str(filter_day).split(',')]
        if len(day_range) == 1:
            # Filter for a single day
            df_filtered = df_filtered[df_filtered['noon_day'] == day_range[0]]
        elif len(day_range) == 2:
            # Filter for a range of days
            df_filtered = df_filtered[(df_filtered['noon_day'] >= day_range[0]) & (df_filtered['noon_day'] <= day_range[1])]

    # Plot trajectory across days for the filtered data
    sns.set(style="whitegrid")
    g = sns.FacetGrid(df_filtered, col="noon_day", col_wrap=4, height=3, aspect=1.5, hue="code", palette="tab10")

    # Plot each animal's trajectory with distinct colors
    g.map(plt.plot, "smoothed_x", "smoothed_y", marker="o", linestyle='-', linewidth=0.5, markersize=0.75)

    # Set titles and labels
    g.set_axis_labels("X Coordinate (meters)", "Y Coordinate (meters)")
    g.set_titles("Day {col_name}")

    # Define arena coordinates in meters
    arena_coords = pd.DataFrame({
        'x': [801.8, 1340, 1580.5, 964.5, 801.8],
        'y': [265.4, 266.8, 1745, 1742.2, 265.4]
    })

    # Convert coordinates from inches to meters
    arena_coords['x'] = arena_coords['x'] * 0.0254
    arena_coords['y'] = arena_coords['y'] * 0.0254

    # Overlay arena boundaries
    for ax in g.axes.flat:
        ax.plot(arena_coords['x'], arena_coords['y'], color='black', linewidth=2, label='Arena Boundary')

    # Adjust legend to show only distinct animal codes
    g.add_legend(title="Animal Code")
    plt.show()
