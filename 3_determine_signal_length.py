import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def find_interval(sig, smooth_len=25, energy_ratio=0.01):
    """
    Endpoints detection of a signal segment.
    
    params:
      sig: input signal
      smooth_len: length of smoothing window for computing energy
      energy_ratio: ratio of peak energy to threshold for finding endpoints
    
    returns:
      left, right: interval endpoints
      Ltrue: true length of the interval
    """
    L = len(sig)

    # 1) Compute energy (smoothed)
    energy = abs(sig)
    energy_smooth = uniform_filter1d(energy, size=smooth_len, mode='nearest')
    
    # 2) Find peak index and energy
    peak_idx = np.argmax(energy_smooth)
    peak_energy = energy_smooth[peak_idx]
    
    # 3) Set threshold for finding endpoints
    threshold = peak_energy * energy_ratio
    
    # 4) Search from peak to left for endpoint
    left = peak_idx
    for i in range(peak_idx, -1, -1):
        if energy_smooth[i] < threshold:
            left = i + 1
            break
    
    # 5) Search from peak to right for endpoint
    right = peak_idx
    for i in range(peak_idx, L):
        if energy_smooth[i] < threshold:
            right = i - 1
            break
    
    return left, right, right - left + 1



# Tool: find the next power of 2
def next_power_of_two(n):
    p = 1
    while p < n:
        p *= 2
    return p

def normalize_segment(sig, left, right, N):
    """
    Crop or pad a segment to length N, and return the normalized signal.
    
    params:
      sig: input signal
      left, right: interval endpoints
      N: desired length
    
    returns:
      normalized signal of length N
    """
    core = sig[left:right+1]
    core_len = len(core)
    
    if core_len >= N:
        # Core segment is already long enough, no need to extend
        start = (core_len - N) // 2
        return core[start:start+N]
    
    # Core segment is too short, need to extend
    need = N - core_len
    
    # Compute how much we can spread to the left and right
    can_left = left  # [0, left) 
    can_right = len(sig) - right - 1  # (right, end] 
    
    # Spread more to the right
    extend_left = min(need // 10, can_left)
    extend_right = min(need - extend_left, can_right)
    
    # Compute new interval: [left-extend_left, right+extend_right]
    new_left = left - extend_left
    new_right = right + extend_right
    extended = sig[new_left:new_right+1]
    
    current_len = len(extended)
    
    if current_len < N:
        # Still not enough, need to pad with zeros
        pad_left = (N - current_len) // 2
        pad_right = N - current_len - pad_left
        result = np.pad(extended, (pad_left, pad_right), mode="constant")
    else:
        result = extended
    
    return result[:N]


if __name__ == "__main__":

    base = "output_auto" 
    out_base = "normalized" 
    os.makedirs(out_base, exist_ok=True)

    print("Task 3: Determine signal length and normalize signals.")

    forces = sorted(os.listdir(base), key=lambda x: int(x))
    all_lengths = []

    # Step 1：统计所有 interval 的真实长度 Ltrue
    for force in forces:
        folder = os.path.join(base, force)
        files = sorted(glob(f"{folder}/interval_*.txt"))

        print(f"{force} N：检测 {len(files)} 个间隔")

        for f in files:
            sig = np.loadtxt(f)

            left, right, Ltrue = find_interval(sig)
            all_lengths.append(Ltrue)

    all_lengths = np.array(all_lengths)

    print(">>> Result")
    print("Min length:", np.min(all_lengths))
    print("Max length:", np.max(all_lengths))
    print("Mean length:", np.mean(all_lengths))
    print("95% quantile:", np.percentile(all_lengths, 95))
    print("99% quantile:", np.percentile(all_lengths, 99))

    # Step 2: Compute reference length L_ref and N as the next power of 2 
    L_ref = int(np.percentile(all_lengths, 99))
    N = next_power_of_two(L_ref)

    print(f"\n👉 Reference length (99% quantile) = {L_ref}")
    print(f"👉 Next power of 2 = {N} points\n")


    # Step 3: Normalize all segments to length N and save to output folder
    for force in forces:
        in_folder = os.path.join(base, force)
        out_folder = os.path.join(out_base, force)
        os.makedirs(out_folder, exist_ok=True)

        files = sorted(glob(f"{in_folder}/interval_*.txt"))

        for f in files:
            sig = np.loadtxt(f)

            left, right, _ = find_interval(sig)
            sigN = normalize_segment(sig, left, right, N)
            # Plot and save normalized signal
            plt.figure(figsize=(12, 4))
            plt.plot(sigN)
            plt.title(f"{f}")
            plt.savefig(os.path.join(out_folder, os.path.basename(f).replace(".txt", f"_{N}.png")))
            plt.close()
            
            outname = os.path.basename(f).replace(".txt", f"_{N}.txt")
            np.savetxt(os.path.join(out_folder, outname), sigN)

    print("Task 3 done.")
