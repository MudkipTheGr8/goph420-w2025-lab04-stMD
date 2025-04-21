# Name: Matthew Davidson UCID: 30182729
# # File Purpose: Gutenberg-Richter law analysis for earthquake data.
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import multi_regress
#
#
# Function Purpose: Linearize the Gutenberg-Richter law for regression analysis.
#
# Parameters:
# - magnitude: array-like, shape = (n,) the vector of earthquake magnitudes
# - count: array-like, shape = (n,) the vector of counts of earthquakes with magnitude >= M
#
# Returns:
# - y: array-like, shape = (n,) the vector of log10(count) values
# - Z: array-like, shape = (n, 2) the design matrix for regression
#
def linearize_gutenberg_richter(magnitude, count):
    n = len(magnitude)
    Z = np.zeros((n, 2))
    Z[:, 0] = 1.0
    Z[:, 1] = -magnitude
    y = np.log10(count)
    return y, Z
#
# Function Purpose: Fit the Gutenberg-Richter law to earthquake data using linear regression.
#
# Parameters:
# - magnitude: array-like, shape = (n,) the vector of earthquake magnitudes
# - count: array-like, shape = (n,) the vector of counts of earthquakes with magnitude >= M
#
# Returns:
# - a_parameter: float the estimated parameter a from the Gutenberg-Richter law
# - b_parameter: float the estimated parameter b from the Gutenberg-Richter law
# - R_squared: float the coefficient of determination, a measure of how well the model fits the data
# - residuals: array-like, shape = (n,) the vector of residuals from the regression
#
def fit_gutenberg_richter(magnitude, count):
    if np.any(count <= 0):
        raise ValueError("Count values must be > 0 for log transformation.")
    y, Z = linearize_gutenberg_richter(magnitude, count)
    a, residuals, R_squared = multi_regress(y, Z)
    a_parameter = a[0]
    b_parameter = a[1]
    return a_parameter, b_parameter, R_squared, residuals
#
# Function Purpose: Plot the Gutenberg-Richter relationship.
#
# Parameters:
# - magnitude: array-like, shape = (n,) the vector of earthquake magnitudes
# - count: array-like, shape = (n,) the vector of counts of earthquakes with magnitude >= M
# - a_parameter: float the estimated parameter a from the Gutenberg-Richter law
# - b_parameter: float the estimated parameter b from the Gutenberg-Richter law
# - interval_name: str optional name for the interval (for title)
#
# Returns:
# - fig: matplotlib figure object containing the plot of the Gutenberg-Richter relationship
#
def plot_gutenberg_richter(magnitudes, counts, a_parameter, b_parameter, interval_name=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(magnitudes) == 0:
        return fig  
    min_mag = np.min(magnitudes)
    max_mag = np.max(magnitudes)
    magnitude_range = np.linspace(min_mag, max_mag, 100)
    N_fit = 10**(a_parameter - b_parameter * magnitude_range)
    ax.scatter(magnitudes, counts, color='blue', label='Data')
    ax.plot(magnitude_range, N_fit, 'r-', label=f'Fit: $N = 10^{{{a_parameter:.2f} - {b_parameter:.2f}M}}$')
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude (M)')
    ax.set_ylabel('$N\ (\geq M)$')
    ax.set_title(f'Gutenberg-Richter Relationship - {interval_name}' if interval_name else 'Gutenberg-Richter Relationship')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    return fig
#
# Function Purpose: Calculate cumulative counts of earthquakes with magnitude >= M.
#
# Parameters:
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
#
# Returns:
# - unique_magnitudes: array-like, shape = (m,) the unique magnitudes rounded to 1 decimal place
# - cumulative_counts: array-like, shape = (m,) the cumulative counts of earthquakes with magnitude >= M
#
def calculate_cumulative_counts(magnitudes, min_magnitude=-1.5):
    if len(magnitudes) == 0:
        return np.array([]), np.array([])
    magnitudes = np.asarray(magnitudes)
    filtered_magnitudes = magnitudes[magnitudes >= min_magnitude]
    if len(filtered_magnitudes) == 0:
        return np.array([]), np.array([])
    magnitude_rounded = np.round(filtered_magnitudes, 1)
    unique_magnitudes = np.sort(np.unique(magnitude_rounded))
    cumulative_counts = np.zeros_like(unique_magnitudes, dtype=int)
    for i, mag in enumerate(unique_magnitudes):
        cumulative_counts[i] = np.sum(magnitude_rounded >= mag)
    return unique_magnitudes, cumulative_counts
#
# Function Purpose: Plot earthquake magnitudes over time.
#
# Parameters:
# - timestamps: array-like, shape = (n,) the vector of timestamps in days since start
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
# - start_time: float optional start time for x-axis
# - end_time: float optional end time for x-axis 
#
# Returns:
# - fig: matplotlib figure object containing the plot of earthquake magnitudes over time
#
def plot_time_series(timestamps, magnitudes, start_time=None, end_time=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(timestamps, magnitudes, alpha=0.7, edgecolor="k")
    ax.set_xlabel('Time (days since start)')
    ax.set_ylabel('Magnitude (M)')
    if start_time and end_time:
        day_start = int(round(start_time))
        day_end = int(round(end_time))
        ax.set_title(f'Time Series of Earthquake Magnitudes: Day {day_start} to Day {day_end}')
        ax.set_xlim(start_time, end_time)
    else:
        ax.set_title("Earthquake Magnitudes Time Series")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig
#
# Function Purpose: Plot daily time series of earthquake magnitudes.
#
# Parameters:
# - time_days: array-like, shape = (n,) the vector of timestamps in days since start
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
#
# Returns:
# - None, but saves individual plots for each day in the current directory
#
def plot_daily_time_series(time_days, magnitudes):
    total_days = int(np.ceil(np.max(time_days)))
    for day in range(total_days):
        start = day
        end = day + 1
        mask = (time_days >= start) & (time_days < end)
        day_time = time_days[mask]
        day_mag = magnitudes[mask]  
        plt.figure()
        plt.scatter(day_time, day_mag, alpha=0.7)
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude (M)')
        plt.title(f'Day {day+1} Time Series')
        plt.grid(True)
        plt.savefig(f'day_{day+1}_time_series.png')
        plt.close()
#
# Function Purpose:
#
# Parameters:
# - results: dictionary containing results for each interval
# - intervals_to_plot: list of interval names to plot
#
# Returns:
# - fig: matplotlib figure object containing the combined plots of Gutenberg-Richter relationships
#
def plot_combined_gr(results, intervals_to_plot):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, interval_name in enumerate(intervals_to_plot):
        if interval_name not in results:
            continue
        result = results[interval_name]
        magnitudes = result['magnitudes']
        counts = result['counts']
        a = result['a_param']
        b = result['b_param']
        ax = axes[i]
        ax.scatter(magnitudes, counts, color='blue', label='Data')
        mag_range = np.linspace(min(magnitudes), max(magnitudes), 100)
        N_fit = 10**(a - b * mag_range)
        ax.plot(mag_range, N_fit, 'r-', label=f'Fit: $a={a:.2f},\ b={b:.2f}$')
        ax.set_yscale('log')
        ax.set_title(interval_name)
        ax.set_xlabel('Magnitude (M)')
        ax.set_ylabel('$N\ (\geq M)$')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    return fig
#
# Function Purpose: Plot combined time series of earthquake magnitudes for all 5 days.
#
# Parameters:
# - timestamps: array-like, shape = (n,) the vector of timestamps in days since start
# - magnitudes: array-like, shape = (n,) the vector of earthquake magnitudes
#
# Returns:
# - fig: matplotlib figure object containing the combined plot of earthquake magnitudes over time
#
def plot_combined_time_series(timestamps, magnitudes):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    days = [
        (0.0, 1.0, "Day 1"),
        (1.0, 2.0, "Day 2"),
        (2.0, 3.0, "Day 3"),
        (3.0, 4.0, "Day 4"),
        (4.0, 5.0, "Day 5")
    ]
    for i, (start, end, name) in enumerate(days):
        ax = axes[i]
        mask = (timestamps >= start) & (timestamps < end)
        day_ts = timestamps[mask] - start
        day_mag = magnitudes[mask]
        hours = day_ts * 24  
        ax.scatter(hours, day_mag, alpha=0.7, edgecolor="k")
        ax.set_title(name)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Magnitude (M)')
        ax.set_ylim(-1.5, 2.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_xlim(0, 24)
    axes[-1].axis('off')
    plt.suptitle('Earthquake Magnitudes Time Series - All Days', y=1.02, fontsize=14)
    plt.tight_layout()
    return fig
