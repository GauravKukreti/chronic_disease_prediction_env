import joblib
import numpy as np

loaded_rf_model = joblib.load('../models/saved/diabetes_rf_model.joblib')
loaded_scaler = joblib.load('../models/saved/diabetes_scaler.joblib')

# input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
input_data_1 = np.array([[2, 85, 66, 29, 0, 26.6, 0.351, 31]])  # No Diabetes
input_data_2 = np.array([[6, 198, 70, 33, 160, 32.2, 0.447, 50]])  # Diabetes

scaled_input_1 = loaded_scaler.transform(input_data_1)
scaled_input_2 = loaded_scaler.transform(input_data_2)

prediction_1 = loaded_rf_model.predict(scaled_input_1)
prediction_proba_1 = loaded_rf_model.predict_proba(scaled_input_1)

prediction_2 = loaded_rf_model.predict(scaled_input_2)
prediction_proba_2 = loaded_rf_model.predict_proba(scaled_input_2)

import os
os.system('cls')

print("\nDIABETES MODEL:\n")
print("Sample data_1\nPregnancies:2, Glucose:85, BloodPressure:66, SkinThickness:29, Insulin:0, BMI:26.6, DiabetesPedigreeFunction:0.351, Age:31\n")
print("Sample data_2\nPregnancies:6, Glucose:198, BloodPressure:70, SkinThickness:33, Insulin:160, BMI:32.2, DiabetesPedigreeFunction:0.447, Age:50\n")

print(f"Prediction for Input: {'Diabetes (1)' if prediction_1[0] == 1 else 'No Diabetes (0)'}")
# print(f"Prediction Probability: No Diabetes (0): {prediction_proba_1[0][0]:.2f}, Diabetes (1): {prediction_proba_1[0][1]:.2f}")
if prediction_proba_1[0][0] > prediction_proba_1[0][1]:
    print(f"with Probability of: {prediction_proba_1[0][0]:.2f}")
else:
    print(f"with Probability of: {prediction_proba_1[0][1]:.2f}")

print(f"\nPrediction for Input: {'Diabetes (1)' if prediction_2[0] == 1 else 'No Diabetes (0)'}")
# print(f"Prediction Probability: No Diabetes (0): {prediction_proba_2[0][0]:.2f}, Diabetes (1): {prediction_proba_2[0][1]:.2f}")
if prediction_proba_2[0][0] > prediction_proba_2[0][1]:
    print(f"with Probability of: {prediction_proba_2[0][0]:.2f}")
else:
    print(f"with Probability of: {prediction_proba_2[0][1]:.2f}")
