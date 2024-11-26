import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pyswarm import pso

df = pd.read_csv('../dataset/processed/heart.csv')

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rf_objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1]) if params[1] > 0 else None
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])
    bootstrap = bool(int(params[4]))

    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, 
                                bootstrap=bootstrap, 
                                random_state=42)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return negative accuracy because PSO minimizes the objective function
    return -accuracy


lb = [50, 1, 2, 1, 0]
ub = [200, 20, 10, 4, 1]

best_params, _ = pso(rf_objective_function, lb, ub, swarmsize=10, maxiter=10)

best_n_estimators = int(best_params[0])
best_max_depth = int(best_params[1]) if best_params[1] > 0 else None
best_min_samples_split = int(best_params[2])
best_min_samples_leaf = int(best_params[3])
best_bootstrap = bool(int(best_params[4]))

print(f"Best parameters found by PSO: n_estimators={best_n_estimators}, max_depth={best_max_depth}, "
      f"min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}, bootstrap={best_bootstrap}")

rf = RandomForestClassifier(n_estimators=best_n_estimators,
                            max_depth=best_max_depth,
                            min_samples_split=best_min_samples_split,
                            min_samples_leaf=best_min_samples_leaf,
                            bootstrap=best_bootstrap, 
                            random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib
joblib.dump(rf, 'saved/heart_disease_model_optimized_with_pso.joblib')
