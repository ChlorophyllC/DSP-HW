import os
import numpy as np
import matplotlib.pyplot as plt

def compute_fft(sig, fs=51200, N=512):
    # 1) 去直流
    sig = sig - np.mean(sig)

    # 2) 加窗
    window = np.hanning(N)
    sig_win = sig * window

    # 3) FFT
    fft_vals = np.fft.rfft(sig_win, n=N)
    amp = np.abs(fft_vals) * 2 / np.sum(window)

    freqs = np.fft.rfftfreq(N, 1/fs)
    return freqs, amp

if __name__ == "__main__":
    base = "normalized" 
    out_base = "fft" 
    os.makedirs(out_base, exist_ok=True)
    for force in os.listdir(base):
        force_dir = os.path.join(base, force)
        if not os.path.isdir(force_dir):
            continue

        # output directories
        out_amp = os.path.join(out_base, force, "amp")
        out_log = os.path.join(out_base, force, "logamp")
        os.makedirs(out_amp, exist_ok=True)
        os.makedirs(out_log, exist_ok=True)

        for fname in os.listdir(force_dir):
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(force_dir, fname)
            sig = np.loadtxt(path).astype(float)

            freqs, amp = compute_fft(sig)
            log_amp = np.log10(amp + 1e-12)

            np.savetxt(os.path.join(out_amp, fname.replace(".txt", "_fft.txt")),
                    np.column_stack([freqs, amp]), fmt="%.6f")

            np.savetxt(os.path.join(out_log, fname.replace(".txt", "_logfft.txt")),
                    np.column_stack([freqs, log_amp]), fmt="%.6f")
            
            # Plot and save FFT result
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.plot(freqs, amp)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.subplot(122)
            plt.plot(freqs, log_amp)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Log Amplitude")
            plt.savefig(os.path.join(out_amp, fname.replace(".txt", "_fft.png")))
            plt.close()

    print("FFT done for all force directories!")
