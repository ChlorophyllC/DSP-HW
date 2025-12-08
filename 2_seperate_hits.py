import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu

def load_signal(txt_file):
    print(f"读取：{txt_file}")
    sig = np.loadtxt(txt_file)
    return sig


# Step1: compute envelope
def compute_envelope(sig, w=30):
    L = len(sig)
    
    env = np.convolve(np.abs(sig), np.ones(w)/w, mode='same')
    return env


# Step2: Auto-thresholding with Otsu's method
def auto_threshold(env):
    
    env_norm = MinMaxScaler().fit_transform(env.reshape(-1,1)).ravel()

    th = threshold_otsu(env_norm)
    print("Threshold =", th)

    return env_norm, th


# Step3: Detect intervals based on threshold
def detect_intervals(env_norm, th):
    L = len(env_norm)
    intervals = []
    active = False
    start = 0

    for i in range(L):
        if env_norm[i] > th and not active:
            active = True
            start = i
        elif env_norm[i] <= th and active:
            active = False
            intervals.append((start, i))

    if active:
        intervals.append((start, L-1))

    print("Initial detected intervals:", len(intervals))
    return intervals


# Step4: Merge intervals based on distance
def merge_intervals(intervals, merge_ratio=0.002):
    """
    merge_ratio：占信号总长的比例
    """
    if not intervals:
        return []

    L = intervals[-1][1]
    min_gap = int(L * merge_ratio)

    merged = []
    cur_l, cur_r = intervals[0]

    for L2, R2 in intervals[1:]:
        if L2 - cur_r < min_gap:
            cur_r = R2
        else:
            merged.append((cur_l, cur_r))
            cur_l, cur_r = L2, R2

    merged.append((cur_l, cur_r))

    print("合并后敲击次数：", len(merged))
    return merged


# Step5: Extract segments based on intervals
def extract_segments(sig, intervals, output_dir, extra_ratio=0.001):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    L = len(sig)
    extra = int(L * extra_ratio)

    for i, (L0, R0) in enumerate(intervals, 1):
        start = max(L0 - extra, 0)
        end = min(R0 + extra, L)

        seg = sig[start:end]

        np.savetxt(f"{output_dir}/interval_{i}.txt", seg)

        plt.figure(figsize=(12, 4))
        plt.plot(seg)
        plt.title(f"Interval {i}")
        plt.savefig(f"{output_dir}/interval_{i}.png")
        plt.close()

    print("All segments saved.")

if __name__ == "__main__":
    forces = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    for force in forces:
        txt = f"槽楔模型测试数据/acquisitionData-{force}.txt"
        out = f"output_auto/{force}"

        sig = load_signal(txt)

        env = compute_envelope(sig)

        env_norm, th = auto_threshold(env)

        intervals = detect_intervals(env_norm, th)

        intervals = merge_intervals(intervals)

        extract_segments(sig, intervals, out)