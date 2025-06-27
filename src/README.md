# House Price Prediction Project

This project implements a machine learning pipeline for predicting house prices using the Ames Housing dataset.

## Project Structure

```
src/
├── train.py      # Training script that creates the model
├── predict.py    # Prediction script for making new predictions
└── model.pkl     # Trained model file (created after running train.py)
```

## Usage

### 1. Train the Model
```bash
cd src
python train.py
```
This will:
- Load and preprocess the data
- Train a Linear Regression model
- Evaluate the model performance
- Save the trained model as `model.pkl`
- Generate a visualization comparing actual vs predicted prices

### 2. Make Predictions

#### View Model Information
```bash
python predict.py --mode info
```

#### Predict with Sample Data
```bash
python predict.py --mode sample
```

#### Interactive Prediction
```bash
python predict.py --mode input
```

## Features Used

The model uses the following features to predict house prices:
- Overall Qual: Overall quality rating
- Gr Liv Area: Above ground living area
- Garage Cars: Size of garage in car capacity
- Total Bsmt SF: Total basement area
- Full Bath: Number of full bathrooms
- Year Built: Year the house was built

## Model Performance

The Linear Regression model achieves:
- R² Score: ~0.78 (78% of variance explained)
- RMSE: ~$42,276
- MAE: ~$25,134

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
