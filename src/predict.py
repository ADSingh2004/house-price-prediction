import pandas as pd
import numpy as np
import pickle
import argparse
import sys

def load_model():
    """Load the trained model and preprocessing components"""
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print("Error: model.pkl not found. Please run train.py first to create the model.")
        sys.exit(1)

def predict_price(features_dict):
    """Predict house price given feature values"""
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['features']
    
    # Create feature array in the correct order
    feature_values = [features_dict[feature] for feature in feature_names]
    feature_array = np.array([feature_values])
    
    # Scale features
    feature_array_scaled = scaler.transform(feature_array)
    
    # Make prediction
    predicted_price = model.predict(feature_array_scaled)
    
    return predicted_price[0]

def predict_from_input():
    """Interactive prediction from user input"""
    model_data = load_model()
    feature_names = model_data['features']
    
    print("Enter values for the following features:")
    features_dict = {}
    
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                features_dict[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    predicted_price = predict_price(features_dict)
    print(f"\nPredicted House Price: ${predicted_price:,.2f}")
    
    return predicted_price

def predict_sample():
    """Predict with a sample house"""
    # Sample house features: [Overall Qual, Gr Liv Area, Garage Cars, Total Bsmt SF, Full Bath, Year Built]
    sample_features = {
        'Overall Qual': 7,
        'Gr Liv Area': 2000,
        'Garage Cars': 2,
        'Total Bsmt SF': 1500,
        'Full Bath': 2,
        'Year Built': 2005
    }
    
    print("Predicting price for sample house:")
    for feature, value in sample_features.items():
        print(f"  {feature}: {value}")
    
    predicted_price = predict_price(sample_features)
    print(f"\nPredicted House Price: ${predicted_price:,.2f}")
    
    return predicted_price

def show_model_info():
    """Display model information and performance metrics"""
    model_data = load_model()
    metrics = model_data['metrics']
    features = model_data['features']
    
    print("Model Information:")
    print("=" * 40)
    print(f"Model Type: Linear Regression")
    print(f"Features used: {', '.join(features)}")
    print("\nModel Performance:")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  MSE: {metrics['mse']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R² Score: {metrics['r2']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='House Price Prediction')
    parser.add_argument('--mode', choices=['sample', 'input', 'info'], default='sample',
                       help='Prediction mode: sample (default), input, or info')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        predict_sample()
    elif args.mode == 'input':
        predict_from_input()
    elif args.mode == 'info':
        show_model_info()

if __name__ == "__main__":
    main()
