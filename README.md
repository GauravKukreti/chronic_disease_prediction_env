
# Chronic Disease Prediction

This repository contains a project aimed at predicting chronic diseases such as Heart Disease and Kidney Disease using machine learning. The models are trained using Random Forest Classifier with hyperparameter optimization (Grid Search and Particle Swarm Optimization) and provide disease predictions based on input medical data.

## Features
- **Disease Prediction**: Predicts the likelihood of chronic diseases based on user inputs.
- **Hyperparameter Optimization**: Utilizes Grid Search and Particle Swarm Optimization for improving model performance.
- **Reusable Models**: Models and scalers are saved for future predictions without retraining.

## Project Workflow
1. **Data Collection**: Acquired datasets for Heart Disease and Kidney Disease with relevant features.
2. **Data Preprocessing**: 
   - Handled missing values, categorical encoding, and feature scaling.
   - Split data into training and testing sets.
3. **Model Training**:
   - Trained Random Forest models with optimized hyperparameters.
   - Saved the trained models and scalers.
4. **Prediction**:
   - Loaded the saved models and scalers for making predictions.
   - Supported both single input and file-based batch predictions.

## How to Run
### Prerequisites
- Python 3.7 or higher
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `joblib`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

### Steps
1. Clone the repository:
```bash
git clone <repository_url>
```
2. Navigate to the project directory:
```bash
cd chronic_disease_prediction
```
3. Run the preprocessing scripts to clean the data.
4. Train the models using the respective scripts in the `src/models` directory.
5. Use the prediction scripts in the `src/predict` directory for making predictions.

### Example Usage
#### Predict from File
Provide input data in `input_kidney.txt` or `input_heart.txt` and run the respective prediction script:
```bash
python src/predict/kidney_predict.py
```

Results will be saved in `output_kidney.txt`.

#### Predict from Code
Use the following Python snippet for prediction:
```python
import numpy as np
from joblib import load

# Load model and scaler
model = load("saved/kidney_disease_model_optimized.joblib")
scaler = load("saved/kidney_scaler_optimized.joblib")

# Input data
input_data = np.array([[70, 140, 1.010, 4, 3, 0, 1, 1, 1, 250, 100, 3.5, 110, 6.2, 7.0, 25, 11000, 2.8, 1, 1, 1, 1, 0, 1]])
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)
print("Prediction:", prediction)
```

## Project Structure
```
chronic_disease_prediction/
├── src/
│   ├── dataset/
│   │  ├── raw             # Raw datasets
│   │  ├── processed/      # Preprocessed datasets
│   ├── models/            # Model training scripts
│   │  ├── saved/          # Saved models and scalers
│   │  ├── preprocess      # Data preprocessing scripts
│   │  ├── model training  # model training scripts
│   ├── predict/           # Prediction scripts
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
