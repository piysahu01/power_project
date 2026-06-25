import os
print("Saving model to: ", os.getcwd())

import pandas as pd
import numpy as np
import os  # <-- Required for the line above
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Simulate Data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'temperature': np.random.normal(30, 5, n),
    'humidity': np.random.normal(50, 10, n),
    'hour_of_day': np.random.randint(0, 24, n),
    'previous_day_usage': np.random.normal(200, 40, n),
})

df['power_usage'] = (
    0.5 * df['temperature'] +
    0.3 * df['humidity'] +
    0.2 * df['previous_day_usage'] +
    np.random.normal(0, 10, n)
)

X = df.drop('power_usage', axis=1)
y = df['power_usage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'power_model.pkl')
print(" Model trained and saved!")
