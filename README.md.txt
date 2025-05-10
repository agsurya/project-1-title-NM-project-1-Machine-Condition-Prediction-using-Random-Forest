
## Machine Condition Prediction using Random Forest

**A.SURYA**
2nd Year – Mechanical Engineering
ARM College of Engineering & Technology
Course: Data Analysis in Mechanical Engineering

---

### About the Project

This project is about predicting the condition of a machine using a **Random Forest Classifier**. The model looks at different parameters like temperature, vibration, oil quality, RPM, and others to decide if the machine is working normally or if there is a fault. It’s a practical use of machine learning in the field of mechanical engineering.

---

### Requirements

To run this project, install the required Python packages by using the command below:

```
pip install -r requirements.txt
```

---

### Files Used in Prediction

* **random\_forest\_model.pkl** – The trained Random Forest model.
* **scaler.pkl** – A Scikit-learn StandardScaler that helps normalize the input values.
* **selected\_features.pkl** – Contains the exact feature names used during training.

All three files should be present in your project folder. If they're missing or not properly linked, the prediction script won’t work correctly.

---

### How the Prediction Works

1. **Loading Files**

   * Load the trained model: `joblib.load('random_forest_model.pkl')`
   * Load the scaler: `joblib.load('scaler.pkl')`
   * Load the feature list: `joblib.load('selected_features.pkl')`

2. **Preparing Input Data**

   * Create a `pandas.DataFrame` with one row of input values.
   * The column names must match the feature names exactly.

3. **Preprocessing**

   * Use the scaler to normalize the input so that it matches the format used during training.

4. **Prediction**

   * Use `.predict()` to get the condition class.
   * Use `.predict_proba()` to see the confidence level for each possible outcome.

---

### How to Make a Prediction

Here’s a sample script you can use to test the model:

```python
import joblib
import pandas as pd

# Load the model, scaler, and features
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input values (replace with real data)
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange columns in the correct order
new_data = new_data[selected_features]

# Scale the input
scaled_data = scaler.transform(new_data)

# Make prediction
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

### Important Tips

* Always use the **same features** as the ones used for training.
* Make sure the **values are realistic** and fall within the expected range.
* **Column order matters**. Do not change it or the model may not work correctly.

---

### Retraining the Model (Optional)

If you want to update the model with new data:

* Use the same steps for preprocessing and feature selection.
* Scale your new data the same way.
* Save the updated model and files using `joblib`.

---

### Real-World Applications

* Checking if machines in factories are in good condition.
* Helping maintenance teams take action before breakdowns.
* Can be used with IoT sensors for real-time diagnostics.

