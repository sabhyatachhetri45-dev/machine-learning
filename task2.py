# Customer Purchase Prediction
# Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




data = {
    'Age': [22, 25, 28, 32, 35, 40, 45, 50, 23, 27,
            31, 36, 42, 48, 52, 29, 34, 39, 44, 55],

    'Income': [25000, 30000, 35000, 45000, 50000, 60000, 70000,
               80000, 28000, 38000, 42000, 55000, 65000, 75000,
               90000, 40000, 48000, 58000, 68000, 95000],

    'Product_Price': [500, 800, 1000, 1500, 2000, 2500, 3000, 3500,
                      600, 1200, 1600, 2200, 2800, 3200, 4000,
                      1300, 1800, 2400, 3000, 4500],

    'Previous_Purchases': [1, 2, 2, 3, 4, 5, 6, 7, 1, 3,
                           4, 5, 6, 7, 8, 2, 4, 5, 6, 9],

    'Buy': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

print("Dataset:")
print(df)




X = df[['Age', 'Income', 'Product_Price', 'Previous_Purchases']]
y = df['Buy']




X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)




model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)




y_pred = model.predict(X_test)




accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)



cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Buy', 'Buy']
)

disp.plot()
plt.title("Customer Purchase Confusion Matrix")
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()




results = X_test.copy()

results['Actual'] = y_test
results['Predicted'] = y_pred

results['Actual'] = results['Actual'].map({
    0: 'Not Buy',
    1: 'Buy'
})

results['Predicted'] = results['Predicted'].map({
    0: 'Not Buy',
    1: 'Buy'
})

results.to_csv('task2_prediction_results.csv', index=False)

print("\nPredicted Examples:")
print(results.head(5))