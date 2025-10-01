# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

 


# Step 2: Create / Load Dataset
np.random.seed(42)
X = np.random.rand(200, 1) * 10          # Feature (e.g., temperature)
y = 3 * X.flatten() + 7 + np.random.randn(200) * 2   # Target (e.g., power consumption)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(kernel='linear'),
    "Decision Tree": DecisionTreeRegressor(max_depth=5,           # limit how deep tree goes
                        min_samples_split=10,  # minimum samples needed to split
                        min_samples_leaf=5,    # minimum samples in leaf node
                        random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Step 5: Train & Evaluate
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MSE": mse, "R2 Score": r2}

    print(f"--- {name} ---")
    print("MSE:", mse)
    print("R2 Score:", r2)
    print()

# Step 6: Compare Results in Bar Chart
mse_scores = [results[m]["MSE"] for m in results]
r2_scores = [results[m]["R2 Score"] for m in results]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.bar(results.keys(), mse_scores, color="orange")
plt.title("Model Comparison - MSE")
plt.ylabel("Mean Squared Error")

plt.subplot(1,2,2)
plt.bar(results.keys(), r2_scores, color="green")
plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.show()
