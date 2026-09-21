# ==========================================
# STUDENT PERFORMANCE PREDICTION
# Machine Learning with Decision Tree
# ==========================================

# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 2. LOAD THE DATA
# ==========================================

data = pd.read_csv("students.csv")

print("========== ORIGINAL DATA ==========")
print(data)


# ==========================================
# 3. UNDERSTAND THE DATA
# ==========================================

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATA INFORMATION ==========")
data.info()

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# ==========================================
# 4. PREPARE / CLEAN THE DATA
# ==========================================

# Remove rows containing missing values
data = data.dropna()

print("\n========== DATA AFTER CLEANING ==========")
print(data)


# ==========================================
# 5. SELECT FEATURES (X) AND TARGET (Y)
# ==========================================

# Features used to make the prediction
X = data[["Age", "StudyHours", "Attendance"]]

# Target we want to predict
y = data["Result"]

print("\n========== FEATURES (X) ==========")
print(X)

print("\n========== TARGET (y) ==========")
print(y)


# ==========================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAINING DATA ==========")
print(X_train)

print("\n========== TESTING DATA ==========")
print(X_test)


# ==========================================
# 7. CREATE THE MACHINE LEARNING MODEL
# ==========================================

model = DecisionTreeClassifier(random_state=42)


# ==========================================
# 8. TRAIN THE MODEL
# ==========================================

model.fit(X_train, y_train)

print("\n========== MODEL TRAINING COMPLETE ==========")
print("The Decision Tree model has been trained.")


# ==========================================
# 9. MAKE PREDICTIONS ON TEST DATA
# ==========================================

y_pred = model.predict(X_test)

print("\n========== PREDICTIONS ==========")
print(y_pred)

print("\n========== ACTUAL RESULTS ==========")
print(y_test.values)


# ==========================================
# 10. CHECK MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))


# ==========================================
# 12. PREDICT A NEW STUDENT
# ==========================================

new_student = pd.DataFrame({
    "Age": [20],
    "StudyHours": [6],
    "Attendance": [92]
})

prediction = model.predict(new_student)

print("\n========== NEW STUDENT ==========")
print(new_student)

print("\n========== PREDICTION ==========")
print("Predicted Result:", prediction[0])