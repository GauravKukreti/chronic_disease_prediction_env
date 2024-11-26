import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('../dataset/kidney_syn.csv')

# df = df.drop(columns=['id'])

print("Missing values per column:\n", df.isnull().sum())

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['classification'])
y = df['classification']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

processed_df = pd.DataFrame(X, columns=df.drop(columns=['classification']).columns)
processed_df['classification'] = y
processed_df.to_csv('../dataset/processed/kidney_syn.csv', index=False)

print("Preprocessing complete and data saved!")
