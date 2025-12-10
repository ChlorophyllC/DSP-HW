import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

def extract_features(freq, amp, force_label):
    amp_smooth = median_filter(amp, size=5)

    def compute_band(fe, am):
        total_energy = np.sum(am ** 2)

        centroid = np.sum(fe * am) / (np.sum(am) + 1e-12)
        bandwidth = np.sqrt(
            np.sum(((fe - centroid) ** 2) * am) / (np.sum(am) + 1e-12)
        )

        psd = am / (np.sum(am) + 1e-12)
        spectral_entropy = -np.sum(psd * np.log(psd + 1e-12))

        peaks, _ = find_peaks(am, prominence=np.max(am) * 0.05)

        peak_freqs = np.zeros(peak_num)
        peak_amps = np.zeros(peak_num)

        top = sorted(peaks, key=lambda p: am[p], reverse=True)[:peak_num]
        for i, p in enumerate(top):
            peak_freqs[i] = fe[p]
            peak_amps[i] = am[p]

        main_peak_freq = peak_freqs[0]
        main_peak_amp = peak_amps[0]

        return {
            "total_energy": total_energy,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "spectral_entropy": spectral_entropy,
            "main_peak_freq": main_peak_freq,
            "main_peak_amp": main_peak_amp,
            "peak_freqs": peak_freqs,
            "peak_amps": peak_amps,
        }

    # ===== Frequency band segmentation =====
    mask_low = (freq < 8000) & (freq >= 100)

    mask_high = (freq >= 8000) & (freq < 20000)

    band_full = compute_band(freq, amp_smooth)
    band_low = compute_band(freq[mask_low], amp_smooth[mask_low])
    band_high = compute_band(freq[mask_high], amp_smooth[mask_high])

    # Assemble feature dictionary
    feat = {"force": force_label}

    def add(prefix, data):
        feat[f"{prefix}_total_energy"] = data["total_energy"]
        feat[f"{prefix}_centroid"] = data["centroid"]
        feat[f"{prefix}_bandwidth"] = data["bandwidth"]
        feat[f"{prefix}_spectral_entropy"] = data["spectral_entropy"]
        feat[f"{prefix}_main_peak_freq"] = data["main_peak_freq"]
        feat[f"{prefix}_main_peak_amp"] = data["main_peak_amp"]

        for i in range(peak_num):
            feat[f"{prefix}_peak{i+1}_freq"] = data["peak_freqs"][i]
            feat[f"{prefix}_peak{i+1}_amp"] = data["peak_amps"][i]

    add("full", band_full)
    add("low", band_low)
    add("high", band_high)

    return feat

if __name__ == "__main__":
    peak_num = 5
    base_dir = 'fft'
    out_dir = 'fft_features'
    os.makedirs(out_dir, exist_ok=True)
    features = []

    for force in sorted(os.listdir(base_dir)):
        force_path = os.path.join(base_dir, force)
        amp_path = os.path.join(force_path, "amp")

        if not os.path.isdir(amp_path):
            continue

        for file in os.listdir(amp_path):
            if not file.endswith(".txt"):
                continue

            fpath = os.path.join(amp_path, file)

            try:
                data = np.loadtxt(fpath)
                freq, amp = data[:, 0], data[:, 1]
            except Exception as e:
                print("Read error:", fpath, e)
                continue

            feat = extract_features(freq, amp, int(force))
            features.append(feat)

    df = pd.DataFrame(features)
    df.to_csv(f"{out_dir}/fft_features_all.csv", index=False)
    print(f"Saved: {out_dir}/fft_features_all.csv")

    df_force = df.groupby("force").mean()
    df_force.to_csv(f"{out_dir}/fft_features_by_force.csv")
    print(f"Saved: {out_dir}/fft_features_by_force.csv")


    # =========================================================
    # Print summary statistics
    # =========================================================
    print("\n===== Full-band features (averaged) =====")
    print(df_force[[f"full_{k}" for k in [
        "main_peak_freq","main_peak_amp","total_energy",
        "centroid","bandwidth","spectral_entropy"
    ]]])

    print("\n===== Low-frequency band features (averaged) =====")
    print(df_force[[f"low_{k}" for k in [
        "main_peak_freq","main_peak_amp","total_energy",
        "centroid","bandwidth","spectral_entropy"
    ]]])

    print("\n===== High-frequency band features (averaged) =====")
    print(df_force[[f"high_{k}" for k in [
        "main_peak_freq","main_peak_amp","total_energy",
        "centroid","bandwidth","spectral_entropy"
    ]]])


    # =========================================================
    # Visualization functions
    # =========================================================


    def plot_force_relation(df_force, feature, ylabel):
        plt.figure()
        plt.plot(df_force.index, df_force[feature], marker="o")
        plt.xlabel("Force")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Force")
        plt.grid(True)
        plt.savefig(f"{out_dir}/{feature}.png", dpi=200)
        plt.close()


    # Plot all combinations: three bands × six core features
    for prefix in ["full", "low", "high"]:
        for feature in ["main_peak_freq","main_peak_amp","total_energy",
                        "centroid","bandwidth","spectral_entropy"]:
            plot_force_relation(
                df_force,
                f"{prefix}_{feature}",
                f"{prefix.capitalize()} {feature.replace('_', ' ').title()}"
            )


    # =========================================================
    # Heatmap of full-band peak frequencies
    # =========================================================
    plt.figure(figsize=(8, 6))
    peak_matrix = df_force[[f"full_peak{i}_freq" for i in range(1, peak_num + 1)]].values
    plt.imshow(peak_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="Frequency (Hz)")
    plt.yticks(range(len(df_force.index)), df_force.index)
    plt.xticks(range(peak_num), [f"Peak {i}" for i in range(1, peak_num + 1)])
    plt.title("Full-Band Peak Frequencies Heatmap")
    plt.savefig(f"{out_dir}/full_peak_freq_heatmap.png", dpi=200)
    plt.close()

    print(f"All plots saved to {out_dir}/ directory")