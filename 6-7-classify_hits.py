import os
import numpy as np
from glob import glob

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Step 1: Load FFT Data ------------------
def load_fft_dataset(base="fft"):
    X = []
    y = []

    forces = sorted(os.listdir(base), key=lambda x: int(x))

    for force in forces:
        folder = os.path.join(base, force)
        files = sorted(glob(f"{folder}/amp/interval_*_fft.txt"))

        for f in files:
            data = np.loadtxt(f)

            # data: [freq, amp]
            amps = data[:,1]

            X.append(amps)
            y.append(int(force))

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    return X, y



# ------------------ Step 2: PCA ------------------
def apply_pca(X, n_components=50):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)

    print(f"PCA → {n_components} dims, variance retained = "
          f"{np.sum(pca.explained_variance_ratio_):.3f}")

    return X_pca, scaler, pca



# ------------------ Step 3: Train SVM + Evaluate ------------------
def train_and_eval(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = SVC(kernel='linear', C=50)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"\n[Seed {random_state}] Accuracy = {acc*100:.2f}%\n")

    print("=== Per-Class Accuracy ===")
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (seed={random_state})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/confusion_matrix.png", dpi=150)
    plt.close()

    return clf, X_train, X_test, y_train, y_test, pred



# ------------------ Visualization 1: PCA Scatter ---------------
def plot_pca_scatter(X_pca, y):
    plt.figure(figsize=(7,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="tab10", s=20)
    plt.title("PCA Scatter (First 2 Components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Force")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/pca_scatter.png", dpi=150)
    plt.close()



# ------------------ Visualization 2: PCA Explained Variance ---------------
def plot_pca_variance(pca):
    plt.figure(figsize=(7,4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title("PCA Explained Variance (Cumulative)")
    plt.xlabel("Components")
    plt.ylabel("Variance Ratio")
    plt.grid()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/pca_variance.png", dpi=150)
    plt.close()



# ------------------ Visualization 3: Mean FFT Spectrum per force ---------------
def plot_mean_spectrum(X, y):
    forces = np.unique(y)
    plt.figure(figsize=(9,6))

    for f in forces:
        mean_amp = np.mean(X[y==f], axis=0)
        plt.plot(mean_amp, label=f"Force {f}")

    plt.title("Mean Spectrum per Force")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/mean_spectrum.png", dpi=150)
    plt.close()



# ------------------ Step 4: Multiple seed evaluation ---------------
def evaluate_multiple_seeds(X, y):
    results = []

    for s in range(0, 100, 10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=s, stratify=y
        )
        clf = SVC(kernel="linear", C=50)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"Seed={s}, Accuracy={acc:.4f}")

    print("\nMean accuracy =", np.mean(results))
    print("Std =", np.std(results))



# ------------------ Main ------------------
if __name__ == "__main__":

    print("\n========= Step 1: Load FFT dataset =========\n")
    X, y = load_fft_dataset("fft")

    print("\n========= Step 2: PCA =========\n")
    X_pca, scaler, pca = apply_pca(X, n_components=50)

    print("\n========= Step 3: SVM Train & Evaluate =========\n")
    clf, X_train, X_test, y_train, y_test, pred = train_and_eval(X_pca, y)

    print("\n========= Step 4: Visualization =========")
    plot_pca_scatter(X_pca, y)
    plot_pca_variance(pca)
    plot_mean_spectrum(X, y)

    print("\n========= Step 5: Evaluate Multiple Seeds =========")
    evaluate_multiple_seeds(X_pca, y)

    print("\nAll results saved to ./figures/")
