# Name: Matthew Davidson UCID: 30182729
# File Purpose: Driver code for analyzing earthquake data using the Gutenberg-Richter law.
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import multi_regress
from lab_04.gutenberg_richter import (
    linearize_gutenberg_richter,
    fit_gutenberg_richter,
    plot_gutenberg_richter,
    calculate_cumulative_counts,
    plot_time_series,
    plot_combined_gr,
    plot_combined_time_series
)
#
# Function Purpose: Load provided earthquake data.
#
# Parameters:
# - file_path: str the path to the earthquake data file
#
# Returns:
# - timestamps: array-like, shape = (n,) the vector of timestamps in days
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
#
def load_earthquake_data(file_path):
    data = np.loadtxt(file_path)
    timestamps_hrs = data[:, 0]
    magnitudes = data[:, 1]
    timestamps_days = timestamps_hrs / 24.0
    return timestamps_days, magnitudes
#
# Function Purpose: Filter data by time range.
#
# Parameters:
# - timestamps: array-like, shape = (n,) the vector of timestamps in days
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
# - start_time: float the start time of the interval in days
# - end_time: float the end time of the interval in days
#
# Returns:
# - filtered_timestamps: array-like, shape = (m,) the vector of filtered timestamps in days
# - filtered_magnitudes: array-like, shape = (m,) the vector of filtered earthquake magnitudes
#
def filter_by_time(timestamps, magnitudes, start_time, end_time):
    mask = (timestamps >= start_time) & (timestamps < end_time)
    return timestamps[mask], magnitudes[mask]
#
# Function Purpose: Analyze earthquake data for specified intervals.
#
# Parameters:
# - timestamps: array-like, shape = (n,) the vector of timestamps in days
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
# - intervals: list of tuples, each containing (start_time, end_time, name) for the intervals to analyze
# - min_magnitude: float the minimum magnitude for filtering (default: -1.5)
#
# Returns:
# - results: dict a dictionary containing the analysis results for each interval
#
def analyze_by_intervals(timestamps, magnitudes, intervals, min_magnitude=-1.5):
    results = {}
    for start_time, end_time, name in intervals:
        interval_timestamps, interval_magnitudes = filter_by_time(timestamps, magnitudes, start_time, end_time)
        if len(interval_magnitudes) < 10:
            print(f"Skipping {name}: insufficient events ({len(interval_magnitudes)}).")
            continue
        unique_mags, counts = calculate_cumulative_counts(interval_magnitudes, min_magnitude=min_magnitude)
        if len(unique_mags) < 3:
            print(f"Skipping {name}: insufficient unique magnitudes ({len(unique_mags)}).")
            continue
        min_count_threshold = 3
        valid_indices = counts >= min_count_threshold
        if np.sum(valid_indices) < 3:
            print(f"Skipping {name}: insufficient points after filtering low counts.")
            continue
        filtered_mags = unique_mags[valid_indices]
        filtered_counts = counts[valid_indices]
        a_param, b_param, r_squared, residuals = fit_gutenberg_richter(filtered_mags, filtered_counts)
        results[name] = {
            'magnitudes': filtered_mags,
            'counts': filtered_counts,
            'a_param': a_param,
            'b_param': b_param,
            'r_squared': r_squared,
            'residuals': residuals,
            'event_count': len(interval_magnitudes),
        }
    return results
#
# Function Purpose: Plot the evolution of b-values over time.
#
# Parameters:
# - results: dict a dictionary containing the analysis results for each interval
# - intervals: list of tuples, each containing (start_time, end_time, name) for the intervals to analyze
#
# Returns:
# - fig: matplotlib figure object containing the plot of b-value evolution
#
def plot_b_value_evolution(results, intervals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    midpoints = []
    b_values = []
    r_squared_values = []
    event_counts = []
    labels = []
    for interval in intervals:
        name = interval[2]
        if name == "Overall Period" or name not in results:
            continue
        mid = (interval[0] + interval[1]) / 2
        midpoints.append(mid)
        b_values.append(results[name]['b_param'])
        r_squared_values.append(results[name]['r_squared'])
        event_counts.append(results[name]['event_count'])  # <-- Populate event_counts
        labels.append(name)
    sorted_indices = np.argsort(midpoints)
    midpoints = np.array(midpoints)[sorted_indices]
    b_values = np.array(b_values)[sorted_indices]
    r_squared_values = np.array(r_squared_values)[sorted_indices]
    event_counts = np.array(event_counts)[sorted_indices]  # <-- Sort event_counts
    labels = np.array(labels)[sorted_indices]
    sc = ax1.scatter(
        midpoints, 
        b_values, 
        c=r_squared_values, 
        cmap='viridis', 
        s=event_counts/20,
        alpha=0.7
    )
    ax2.plot(midpoints, r_squared_values, 'o-', alpha=0.7)
    ax2.set_ylabel('R²')
    ax2.set_xlabel('Time (days since start)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    for i, label in enumerate(labels):
        ax1.annotate(
            label, 
            (midpoints[i], b_values[i]), 
            fontsize=8, 
            ha='center', 
            va='bottom'
        )
    ax1.set_ylabel('b-value')
    ax1.set_title('Evolution of Gutenberg-Richter b-value Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('R² value')
    plt.tight_layout()
    return fig
#
# Function Purpose: Main function to load data, analyze intervals, and plot results.
#
# Parameters:
# - None
#
# Returns:
# - Gutenberg-Richter analysis results and plots
# - Displays summary of Gutenberg-Richter parameters for each interval
# - Saves plots to figures directory
#
def main():
    data_file = "../data/earthquake_data.txt"
    try:
        timestamps, magnitudes = load_earthquake_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    min_time = np.floor(np.min(timestamps))
    max_time = np.ceil(np.max(timestamps))
    total_days = int(np.ceil(max_time - min_time))
    fig = plot_time_series(timestamps, magnitudes)
    plt.savefig("../figures/overall_time_series.png")
    plt.close(fig)
    key_intervals = [
        (min_time, max_time, "Overall Period"),
        # Daily intervals
        (0.0, 1.0, "Day 1"),
        (1.0, 2.0, "Day 2"),
        (2.0, 3.0, "Day 3"),
        (3.0, 4.0, "Day 4"),
        (4.0, 5.0, "Day 5"),
    ]
    results = analyze_by_intervals(timestamps, magnitudes, key_intervals)
    b_value_fig = plot_b_value_evolution(results, key_intervals)
    plt.savefig("../figures/b_value_evolution.png")
    plt.close(b_value_fig)
    print("\nGutenberg-Richter Parameter Summary:")
    print("=" * 70)
    print(f"{'Interval':<20} {'Events':<8} {'a':<8} {'b':<8} {'R²':<8} {'Mag Range':<15}")
    print("-" * 70)
    for name, result in sorted(results.items(), 
                            key=lambda x: x[1]['b_param'], 
                            reverse=True):  # Sort by b-value
        mag_range = f"{np.min(result['magnitudes']):.1f}-{np.max(result['magnitudes']):.1f}"
        print(f"{name:<20} {result['event_count']:<8d} {result['a_param']:<8.2f} "
              f"{result['b_param']:<8.2f} {result['r_squared']:<8.2f} {mag_range:<15}")
    intervals_to_plot = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Overall Period"]
    combined_fig = plot_combined_gr(
        {k: v for k, v in results.items() if k in intervals_to_plot}, 
        intervals_to_plot
    )
    combined_fig.savefig("../figures/combined_gr_analysis.png", dpi=300)
    plt.close(combined_fig)
    data_file = "../data/earthquake_data.txt"
    timestamps, magnitudes = load_earthquake_data(data_file)
    combined_ts_fig = plot_combined_time_series(timestamps, magnitudes)
    combined_ts_fig.savefig("../figures/combined_time_series.png", dpi=300)
    plt.close(combined_ts_fig)

if __name__ == "__main__":
    main()
