import numpy as np
import joblib

model = joblib.load("../models/saved/kidney_model.joblib")
scaler = joblib.load("../models/saved/kidney_scaler.joblib")

# input_data_1 = np.array([[50, 80, 1.020, 0, 0, 1, 1, 0, 0, 140, 36, 1.2, 135, 4.5, 15.8, 44, 7800, 5.2, 1, 1, 0, 1, 0, 0]])
# input_data_1 = np.array([[73.0,90.0,1.015,3.0,0.0,1,0,1,0,107.0,33.0,1.5,141.0,4.6,10.1,18,72,20,0,3,1,1,0,0]])

#No Kidney Disease
input_data_1 = np.array([[43.0,80.0,1.02,0.0,0.0,1,1,0,0,114.0,32.0,1.1,135.0,3.9,12.649999999999999,30,90,34,0,3,1,0,0,0]])

#Kidney Disease
input_data_extreme = np.array([[70, 140, 1.010, 4, 3, 0, 1, 1, 1, 250, 100, 3.5, 110, 6.2, 7.0, 25, 11000, 2.8, 1, 1, 1, 1, 0, 1]])


input_data_scaled = scaler.transform(input_data_1)
input_data_scaled_extreme = scaler.transform(input_data_extreme)

prediction = model.predict(input_data_scaled)
prediction_prob = model.predict_proba(input_data_scaled)

prediction_extreme = model.predict(input_data_scaled_extreme)
prediction_prob_extreme = model.predict_proba(input_data_scaled_extreme)


class_labels = ["No Kidney Disease", "Kidney Disease"]
predicted_class = prediction[0]
predicted_class_label = class_labels[predicted_class]

predicted_class_extreme = prediction_extreme[0]
predicted_class_label_extreme = class_labels[predicted_class_extreme]

import os
os.system('cls')

print("\nKIDNEY MODEL:\n")
print("Sample data_1\nage:43.0, bp:80.0, sg:1.02, al:0.0, su:0.0, rbc:1, pc:1, pcc:0, ba:0, bgr:114.0, bu:32.0, sc:1.1, sod:135.0, pot:3.9, hemo:12.649999999999999, pcv:30, wc:90, rc:34, htn:0, dm:3, cad:1, appet:0, pe:0, ane:0\n")

print("Sample data_2\nage:70, bp:140, sg:1.010, al:4, su:3, rbc:0, pc:1, pcc:1, ba:1, bgr:250, bu:100, sc:3.5, sod:110, pot:6.2, hemo:7.0, pcv:25, wc:11000, rc:2.8, htn:1, dm:1, cad:1, appet:1, pe:0, ane:1\n")

print(f"\nPrediction for Input 1: {predicted_class_label} ({predicted_class})")
# print(f"Prediction Probability: No Kidney Disease (0): {prediction_prob[0][0]:.2f}, Kidney Disease (1): {prediction_prob[0][1]:.2f}")
if prediction_prob[0][0] > prediction_prob[0][1]:
    print(f"with a probability of: {prediction_prob[0][0]:.2f}")
else:
    print(f"with a probability of: {prediction_prob[0][1]:.2f}")

print(f"\nPrediction for Input 2: {predicted_class_label_extreme} ({predicted_class_extreme})")
# print(f"Prediction Probability: No Kidney Disease (0): {prediction_prob_extreme[0][0]:.2f}, Kidney Disease (1): {prediction_prob_extreme[0][1]:.2f}")
if prediction_prob_extreme[0][0] > prediction_prob_extreme[0][1]:
    print(f"with a probability of: {prediction_prob_extreme[0][0]:.2f}")
else:
    print(f"with a probability of: {prediction_prob_extreme[0][1]:.2f}")