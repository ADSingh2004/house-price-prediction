import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib to avoid display issues
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_and_preprocess_data():
    """Load and preprocess the housing data"""
    # Load data
    df = pd.read_csv("../data/AmesHousing.csv")
    
    # Define features and target
    features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Full Bath', 'Year Built']
    target = 'SalePrice'
    
    # Clean data
    df = df[features + [target]].dropna()
    
    return df, features, target

def train_model():
    """Train the linear regression model"""
    print("Loading and preprocessing data...")
    df, features, target = load_and_preprocess_data()
    
    # Feature scaling
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Model evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Visualization
    print("Creating visualization...")
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.savefig("../actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization saved as 'actual_vs_predicted.png'")
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'model.pkl'")
    return model_data

if __name__ == "__main__":
    train_model()
