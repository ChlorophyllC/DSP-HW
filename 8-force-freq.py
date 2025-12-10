import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import csv


def load_mean_spectra(base_dir):
    """Load and average amplitude spectra for each force level."""
    force_dirs = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: int(x)
    )
    if not force_dirs:
        raise RuntimeError(f"No force subdirectories found in '{base_dir}'.")

    mean_spectra = []
    frequencies = None
    force_values = []

    for force_str in force_dirs:
        amp_files = sorted(glob(os.path.join(base_dir, force_str, "amp", "*.txt")))
        if not amp_files:
            raise RuntimeError(f"No amplitude files found under force {force_str}")

        amps = np.array([np.loadtxt(f)[:, 1] for f in amp_files])
        mean_amp = np.mean(amps, axis=0)
        smoothed_amp = gaussian_filter1d(mean_amp, sigma=1.0)
        mean_spectra.append(smoothed_amp)

        if frequencies is None:
            frequencies = np.loadtxt(amp_files[0])[:, 0]

        force_values.append(int(force_str))

    return np.array(mean_spectra), np.array(frequencies), np.array(force_values), force_dirs


def plot_spectrum_heatmap(mean_spectra, frequencies, force_dirs, out_dir):
    """Plot and save the overall mean spectrum heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(
        20 * np.log10(mean_spectra.T + 1e-12),
        aspect='auto',
        origin='lower',
        extent=[0, len(force_dirs) - 1, frequencies[0], frequencies[-1]],
        cmap='viridis'
    )
    plt.yticks(np.linspace(frequencies[0], frequencies[-1], 6))
    plt.xticks(range(len(force_dirs)), force_dirs, rotation=45)
    plt.xlabel("Force")
    plt.ylabel("Frequency (Hz)")
    plt.title("Mean Spectrum Heatmap (dB)")
    plt.colorbar(label='Amplitude (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectrum_heatmap.png"), dpi=150)
    plt.close()


def track_peaks_in_band(
    mean_spectra,
    frequencies,
    force_dirs,
    freq_low,
    freq_high,
    prominence_fraction=0.05,
    min_width=1,
    center_frequency=None
):
    """Track dominant spectral peak within a frequency band for each force."""
    idx_low = np.searchsorted(frequencies, freq_low, side='left')
    idx_high = np.searchsorted(frequencies, freq_high, side='right') - 1

    tracked_frequencies = []
    tracked_amplitudes = []
    tracked_forces = []

    for i, force in enumerate(force_dirs):
        spec = mean_spectra[i]
        segment = spec[idx_low:idx_high + 1]

        prominence_threshold = max(segment.max() * prominence_fraction, 1e-12)
        peaks, _ = find_peaks(segment, prominence=prominence_threshold, width=min_width)

        if len(peaks) == 0:
            local_peak_idx = np.argmax(segment)
            global_peak_idx = idx_low + local_peak_idx
        else:
            global_peaks = peaks + idx_low
            if center_frequency is None:
                amplitudes_at_peaks = spec[global_peaks]
                best_idx = np.argmax(amplitudes_at_peaks)
            else:
                distances = np.abs(frequencies[global_peaks] - center_frequency)
                best_idx = np.argmin(distances)
            global_peak_idx = global_peaks[best_idx]

        freq_val = frequencies[global_peak_idx]
        amp_val = spec[global_peak_idx]

        tracked_frequencies.append(freq_val)
        tracked_amplitudes.append(amp_val)
        tracked_forces.append(int(force))

    return (
        np.array(tracked_frequencies),
        np.array(tracked_amplitudes),
        np.array(tracked_forces)
    )


def plot_tracked_peaks_on_heatmap(
    mean_spectra, frequencies, force_dirs, tracked_frequencies, freq_low, freq_high, out_dir
):
    """Overlay tracked peak trajectory on spectrum heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(
        20 * np.log10(mean_spectra.T + 1e-12),
        aspect='auto',
        origin='lower',
        extent=[0, len(force_dirs) - 1, frequencies[0], frequencies[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xticks(range(len(force_dirs)), force_dirs, rotation=45)
    plt.plot(
        np.arange(len(force_dirs)),
        tracked_frequencies,
        'r-o',
        label='Tracked peak'
    )
    plt.title("Mean Spectrum Heatmap with Tracked Peak")
    plt.xlabel("Force")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(freq_low - 500, freq_high + 500)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tracked_peak_on_heatmap.png"), dpi=200)
    plt.close()


def fit_and_plot_models(tracked_forces, tracked_frequencies, out_dir):
    """Fit linear and sqrt(force) models, print metrics, and plot results."""
    # Linear fit: f = a * force + b
    slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(tracked_forces, tracked_frequencies)
    r2_lin = r_lin ** 2
    pred_lin = slope_lin * tracked_forces + intercept_lin
    rmse_lin = np.sqrt(np.mean((tracked_frequencies - pred_lin) ** 2))

    # Square-root fit: f = a * sqrt(force) + b
    sqrt_forces = np.sqrt(tracked_forces)
    slope_sqrt, intercept_sqrt, r_sqrt, p_sqrt, _ = stats.linregress(sqrt_forces, tracked_frequencies)
    r2_sqrt = r_sqrt ** 2
    pred_sqrt = slope_sqrt * sqrt_forces + intercept_sqrt
    rmse_sqrt = np.sqrt(np.mean((tracked_frequencies - pred_sqrt) ** 2))

    # Print summary
    print("\nTracked peak frequencies per force:")
    for f, fr, a in zip(tracked_forces, tracked_frequencies, [0]*len(tracked_forces)):  # amplitude not used here
        print(f"  {f:4d} N -> {fr:8.2f} Hz")

    print("\nLinear fit: f = a * force + b")
    print(f"  a = {slope_lin:.6e}, b = {intercept_lin:.4f}, R² = {r2_lin:.4f}, p = {p_lin:.3e}, RMSE = {rmse_lin:.3f} Hz")

    print("\nSquare-root fit: f = a * sqrt(force) + b")
    print(f"  a = {slope_sqrt:.6e}, b = {intercept_sqrt:.4f}, R² = {r2_sqrt:.4f}, p = {p_sqrt:.3e}, RMSE = {rmse_sqrt:.3f} Hz")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(tracked_forces, tracked_frequencies, label="Measured", color='tab:blue')
    x_smooth = np.linspace(tracked_forces.min(), tracked_forces.max(), 200)
    plt.plot(x_smooth, slope_lin * x_smooth + intercept_lin, 'r-', label=f"Linear fit (R²={r2_lin:.3f})")
    plt.plot(x_smooth, slope_sqrt * np.sqrt(x_smooth) + intercept_sqrt, 'g--', label=f"√Force fit (R²={r2_sqrt:.3f})")
    plt.xlabel("Force")
    plt.ylabel("Tracked Peak Frequency (Hz)")
    plt.title("Tracked Peak vs Force with Model Fits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "peak_vs_force_fit.png"), dpi=200)
    plt.close()

    return {
        "linear": {"slope": slope_lin, "intercept": intercept_lin, "r2": r2_lin, "rmse": rmse_lin, "p": p_lin},
        "sqrt": {"slope": slope_sqrt, "intercept": intercept_sqrt, "r2": r2_sqrt, "rmse": rmse_sqrt, "p": p_sqrt}
    }


def save_results_to_csv(tracked_forces, tracked_frequencies, tracked_amplitudes, out_dir):
    """Save tracked peak data to CSV."""
    csv_path = os.path.join(out_dir, "tracked_peak_values.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["force", "freq_Hz", "amplitude"])
        for f, fr, a in zip(tracked_forces, tracked_frequencies, tracked_amplitudes):
            writer.writerow([f, fr, a])
    print(f"\nAll outputs saved to '{out_dir}'")


def main():
    # Configuration
    base_dir = "fft"
    out_dir = "force_freq_ana"
    os.makedirs(out_dir, exist_ok=True)

    FREQ_LOW = 4000
    FREQ_HIGH = 8000
    PROMINENCE_FRACTION = 0.05
    MIN_PEAK_WIDTH = 1
    SMOOTHING_SIGMA = 1.0  # Note: hardcoded in load_mean_spectra; could be parameterized
    CENTER_FREQUENCY = None  # e.g., 5000.0

    # Step 1: Load data
    mean_spectra, frequencies, force_values, force_dirs = load_mean_spectra(base_dir)

    # Step 2: Plot overall heatmap
    plot_spectrum_heatmap(mean_spectra, frequencies, force_dirs, out_dir)

    # Step 3: Track peaks
    tracked_frequencies, tracked_amplitudes, tracked_forces = track_peaks_in_band(
        mean_spectra=mean_spectra,
        frequencies=frequencies,
        force_dirs=force_dirs,
        freq_low=FREQ_LOW,
        freq_high=FREQ_HIGH,
        prominence_fraction=PROMINENCE_FRACTION,
        min_width=MIN_PEAK_WIDTH,
        center_frequency=CENTER_FREQUENCY
    )

    # Step 4: Plot heatmap with tracked peaks
    plot_tracked_peaks_on_heatmap(
        mean_spectra, frequencies, force_dirs, tracked_frequencies, FREQ_LOW, FREQ_HIGH, out_dir
    )

    # Step 5 & 6: Fit models and plot
    fit_and_plot_models(tracked_forces, tracked_frequencies, out_dir)

    # Step 7: Save to CSV
    save_results_to_csv(tracked_forces, tracked_frequencies, tracked_amplitudes, out_dir)


if __name__ == "__main__":
    main()