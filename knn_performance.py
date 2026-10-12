import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Experiment settings
DATASET_SIZES = [500, 2000, 5000, 10000]
ALGORITHMS = ["brute", "kd_tree", "ball_tree"]

results = []

# 2. Generate datasets of different sizes
for size in DATASET_SIZES:
    print(f"\nTesting dataset size: {size}")

    X, y = make_classification(
        n_samples=size,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # 3. Train and test each search algorithm
    for algorithm in ALGORITHMS:
        knn = KNeighborsClassifier(
            n_neighbors=5,
            algorithm=algorithm,
            metric="euclidean",
            n_jobs=-1,
        )

        start_train = time.perf_counter()
        knn.fit(X_train, y_train)
        training_time = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        y_pred = knn.predict(X_test)
        prediction_time = time.perf_counter() - start_predict

        accuracy = accuracy_score(y_test, y_pred)

        results.append(
            {
                "Dataset Size": size,
                "Algorithm": algorithm,
                "Training Time (s)": training_time,
                "Prediction Time (s)": prediction_time,
                "Accuracy": accuracy,
            }
        )

# 4. Display runtime table
results_df = pd.DataFrame(results)

print("\n===== KNN PERFORMANCE RESULTS =====")
print(
    results_df.to_string(
        index=False,
        formatters={
            "Training Time (s)": "{:.6f}".format,
            "Prediction Time (s)": "{:.6f}".format,
            "Accuracy": "{:.4f}".format,
        },
    )
)

results_df.to_csv("knn_results.csv", index=False)

# 5. Plot prediction time versus dataset size
plt.figure(figsize=(9, 6))

for algorithm in ALGORITHMS:
    subset = results_df[results_df["Algorithm"] == algorithm]
    plt.plot(
        subset["Dataset Size"],
        subset["Prediction Time (s)"],
        marker="o",
        linewidth=2,
        label=algorithm,
    )

plt.xlabel("Dataset Size (Number of Samples)")
plt.ylabel("Prediction Time (Seconds)")
plt.title("KNN Prediction Time vs Dataset Size")
plt.legend(title="Search Algorithm")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("knn_prediction_time.png", dpi=300)
plt.close()

# 6. Compare average prediction times
print("\n===== AVERAGE PREDICTION TIME =====")
average_times = results_df.groupby("Algorithm")["Prediction Time (s)"].mean()
print(average_times)

# 7. Find the fastest algorithm for each dataset size
print("\n===== FASTEST ALGORITHM BY DATASET SIZE =====")
for size in DATASET_SIZES:
    subset = results_df[results_df["Dataset Size"] == size]
    fastest = subset.loc[subset["Prediction Time (s)"].idxmin()]
    print(
        f"{size} samples: {fastest['Algorithm']} "
        f"({fastest['Prediction Time (s)']:.6f} seconds)"
    )
