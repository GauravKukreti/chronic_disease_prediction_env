import joblib
import numpy as np

model = joblib.load('../models/saved/heart_model.joblib')
scaler = joblib.load('../models/saved/heart_scaler.joblib')

input_data_1 = np.array([[67,1,0,160,286,0,0,108,1,1.5,1,3,2]]) #No Heart Disease
input_data_2 =np.array([[38,1,2,138,175,0,1,173,0,0.0,2,4,2]]) #Heart Disease

scaled_input_1 = scaler.transform(input_data_1)
scaled_input_2 = scaler.transform(input_data_2)

prediction_1 = model.predict(scaled_input_1)
prediction_2 = model.predict(scaled_input_2)

import os
os.system('cls')

print("\nHEART MODEL:\n")
print("Sample data_1\nage:67, sex:1, cp:0, trestbps:160, chol:286, fbs:0, restecg:0, thalach:108, exang:1, oldpeak:1.5, slope:1, ca:3, thal:2\n")
print("Sample data_2\nage:38, sex:1, cp:2, trestbps:138, chol:175, fbs:0, restecg:1, thalach:173, exang:0, oldpeak:0.0, slope:2, ca:4, thal:2\n")


print(f"Prediction for Input 1: {'Heart Disease (1)' if prediction_1 == 1 else 'No Heart Disease (0)'}")
print(f"Prediction Probability for Input 1: No Heart Disease (0): {model.predict_proba(scaled_input_1)[0][0]:.2f}, Heart Disease (1): {model.predict_proba(scaled_input_1)[0][1]:.2f}")


print(f"\nPrediction for Input 2: {'Heart Disease (1)' if prediction_2 == 1 else 'No Heart Disease (0)'}")
print(f"Prediction Probability for Input 2: No Heart Disease (0): {model.predict_proba(scaled_input_2)[0][0]:.2f}, Heart Disease (1): {model.predict_proba(scaled_input_2)[0][1]:.2f}")