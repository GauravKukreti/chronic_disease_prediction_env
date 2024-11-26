import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = '../dataset/diabetes.csv'
df = pd.read_csv(file_path)

print("Dataset Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

# Convert categorical columns to numeric (if any)
# For example, if there are binary columns, convert them to 0 and 1
# df['column_name'] = df['column_name'].map({'Yes': 1, 'No': 0})

# Or for multi-class categorical columns, use one-hot encoding
# df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Outcome'])
y = df['Outcome']

print("\nFeature Data Preview:")
print(X.head())
print("\nTarget Data Preview:")
print(y.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)

df.to_csv('../dataset/processed/diabetes.csv', index=False)
