import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib to avoid display issues in some environments
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("data/AmesHousing.csv")

# 2. Data Cleaning & Preprocessing
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Full Bath', 'Year Built']
target = 'SalePrice'
df = df[features + [target]].dropna()

# 3. Feature Selection (already chosen in this example)

# 4. Feature Scaling
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 7. Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("actual_vs_predicted.png")
plt.show()

# 8. Make Prediction on New Data
new_house = np.array([[7, 2000, 2, 1500, 2, 2005]])
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
