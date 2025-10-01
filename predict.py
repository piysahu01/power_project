import joblib
import numpy as np

# Load the trained model
model = joblib.load('C:/Users/Piyush Sahu/power_model.pkl')  # 👈 Full path to your saved file

# Simulate new input data:
# Format: [temperature, humidity, hour_of_day, previous_day_usage]
new_input = np.array([[31.5, 58.2, 16, 210.0]])

# Make prediction
prediction = model.predict(new_input)
print("⚡ Predicted Power Usage:", prediction[0])
