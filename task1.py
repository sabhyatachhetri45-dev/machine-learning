import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------
# 1. Load or create dataset
# ------------------------------------------

CSV_PATH = "calories.csv"
os.makedirs("graphs", exist_ok=True)

if not os.path.exists(CSV_PATH):
    rng = np.random.default_rng(42)
    n = 250

    exercise_duration = rng.uniform(20, 90, n)
    age = rng.integers(18, 65, n)
    weight = rng.uniform(45, 110, n)
    heart_rate = rng.uniform(60, 100, n)
    intensity = rng.uniform(30, 95, n)

    calories = (
        80
        + 2.8 * exercise_duration
        + 0.6 * weight
        + 1.1 * heart_rate
        + 0.7 * intensity
        - 0.3 * age
        + rng.normal(0, 10, n)
    )

    df = pd.DataFrame({
        "Exercise Duration": exercise_duration,
        "Age": age,
        "Weight": weight,
        "Heart Rate": heart_rate,
        "Exercise Intensity": intensity,
        "Calories Burned": calories,
    })

    df.to_csv(CSV_PATH, index=False)
    print("Created a synthetic dataset: calories.csv")
else:
    df = pd.read_csv(CSV_PATH)
    print("Loaded dataset from calories.csv")

print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------
# 2. Features and target
# ------------------------------------------

features = [
    "Exercise Duration",
    "Age",
    "Weight",
    "Heart Rate",
    "Exercise Intensity",
]
target = "Calories Burned"

X = df[features]
y = df[target]


# ------------------------------------------
# 3. Different graph styles
# ------------------------------------------

# Bar chart: feature importance / correlation with target
correlation_values = df[features + [target]].corr()[target].drop(target)

plt.figure(figsize=(9, 6))
bar_colors = sns.color_palette("viridis", len(correlation_values))
plt.bar(correlation_values.index, correlation_values.values, color=bar_colors)
plt.title("Feature Correlation with Calories Burned")
plt.xlabel("Feature")
plt.ylabel("Correlation")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("graphs/feature_correlation_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

# Box plot: calories burned by exercise intensity groups
bins = pd.qcut(df["Exercise Intensity"], q=4, labels=["Low", "Medium", "High", "Very High"])
df_plot = df.copy()
df_plot["Intensity Group"] = bins

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_plot, x="Intensity Group", y="Calories Burned", palette="Set2")
plt.title("Calories Burned by Exercise Intensity Group")
plt.xlabel("Exercise Intensity Group")
plt.ylabel("Calories Burned")
plt.tight_layout()
plt.savefig("graphs/calories_by_intensity_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------
# 4. Train model
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel performance:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")


# ------------------------------------------
# 5. Save an additional prediction plot
# ------------------------------------------

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.8)
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="red")
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted Calories Burned")
plt.tight_layout()
plt.savefig("graphs/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved graphs:")
for name in sorted(os.listdir("graphs")):
    print("-", name)

print("\nProject completed successfully.")


# ============================================================
# 14. CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. CALCULATE R-SQUARED
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 16. DISPLAY MODEL RESULTS
# ============================================================

print("\n==============================================")
print("          MODEL PERFORMANCE")
print("==============================================")

print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 4))


# ============================================================
# 17. DISPLAY REGRESSION EQUATION
# ============================================================

print("\n==============================================")
print("          REGRESSION EQUATION")
print("==============================================")

print("Intercept:", round(model.intercept_, 4))

print("\nCoefficients:")

for feature, coefficient in zip(
    features,
    model.coef_
):

    print(
        feature,
        ":",
        round(coefficient, 4)
    )


# ============================================================
# 18. COEFFICIENT ANALYSIS
# ============================================================

coefficient_table = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\n==============================================")
print("          COEFFICIENT ANALYSIS")
print("==============================================")

print(coefficient_table)


# Find strongest coefficient
strongest_index = np.argmax(
    np.abs(model.coef_)
)

strongest_feature = features[strongest_index]

strongest_coefficient = model.coef_[strongest_index]


print("\nStrongest coefficient:")
print(strongest_feature)

print(
    "Coefficient:",
    round(strongest_coefficient, 4)
)


# ============================================================
# 19. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

# Perfect prediction line
minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")

plt.title(
    "Actual vs Predicted Calories Burned"
)

plt.tight_layout()

plt.savefig(
    "graphs/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. DISPLAY SOME PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Calories": y_test.values,
    "Predicted Calories": y_pred
})

print("\n==============================================")
print("          ACTUAL VS PREDICTED")
print("==============================================")

print(results.head(10))


# ============================================================
# 21. SAVE PREDICTION RESULTS
# ============================================================

results.to_csv(
    "prediction_results.csv",
    index=False
)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("             PROJECT COMPLETED")
print("==============================================")

print("\nGraphs have been saved inside the 'graphs' folder.")

print("\nCreated files:")

print("1. Exercise Duration vs Calories")
print("2. Age vs Calories")
print("3. Weight vs Calories")
print("4. Heart Rate vs Calories")
print("5. Exercise Intensity vs Calories")
print("6. Correlation Matrix")
print("7. Actual vs Predicted")

print("\nPrediction results saved as:")
print("prediction_results.csv")
plt.savefig("graphs/your_plot.png", dpi=300, bbox_inches="tight")
plt.close()
