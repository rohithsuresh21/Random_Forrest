import pandas as pd              # For data manipulation
from sklearn.model_selection import train_test_split         # For splitting the dataset
from sklearn.ensemble import RandomForestRegressor       # For Random Forest model
from sklearn.metrics import mean_absolute_error             # For model evaluation

# Sample dataset
data = {
    'sqft': [1500, 2000, 1200, 2500, 1800, 1350, 3000, 2200],
    'bedrooms': [3, 3, 2, 4, 3, 2, 5, 3],
    'age_of_house': [10, 5, 20, 1, 15, 12, 2, 8],
    'price': [150, 210, 110, 320, 175, 125, 400, 230]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('price', axis=1)               # Features
y = df['price']                            # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)            # Initialize model, estimators=100, random_state=42 (for reproducibility)
model.fit(X_train, y_train)               # Train model

# Prediction
y_pred = model.predict(X_test)              # Predict on test set

# Evaluation
mae = mean_absolute_error(y_test, y_pred)               # Calculate Mean Absolute Error  
print("Mean Absolute Error:", mae)       

# Predict new house
new_house = [[2100, 3, 4]]
predicted_price = model.predict(new_house)               #

print(f"Predicted Price: ₹{predicted_price[0]:.2f} lakhs")
