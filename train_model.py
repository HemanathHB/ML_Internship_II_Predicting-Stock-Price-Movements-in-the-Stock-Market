import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset
df = pd.read_csv('stock_data.csv')  # Replace with your dataset path

# Create the target variable
df['PriceMovement'] = df['ClosePrice'].shift(-1)  # Shift the ClosePrice to create labels
df['PriceMovement'] = df.apply(lambda row: 'Up' if row['PriceMovement'] > row['ClosePrice'] else 'Down', axis=1)

# Drop the last row as it won't have a target
df = df[:-1]

# Define features and target
X = df.drop(['PriceMovement', 'Date', 'StockID'], axis=1)
y = df['PriceMovement']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model pipeline
model = Pipeline(steps=[
    ('encoder', ColumnTransformer(
        transformers=[
            ('num', 'passthrough', X_train.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', OneHotEncoder(), ['Sector'])  # Assuming 'Sector' is categorical
        ])),
    ('classifier', RandomForestClassifier())
])

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'stock_price_model.pkl')
print("Model trained and saved as 'stock_price_model.pkl'")
