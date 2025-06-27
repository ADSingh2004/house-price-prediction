# House Price Prediction

A machine learning project to predict house prices using the Ames Housing Dataset, built with Python, Scikit-Learn, and Pandas.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [License](#license)
- [Author](#author)

---

## Project Overview

This project develops a machine learning model to predict the sale price of houses from the Ames Housing Dataset. The primary goal is to build a reliable predictive model by following a standard data science workflow, including data preprocessing, feature engineering, model training, and evaluation.

The initial model is a Linear Regression baseline, with the project structured to allow for easy integration of more advanced algorithms like Random Forest or XGBoost in the future.

---

## Technology Stack

- **Programming Language:** Python
- **Libraries:**
    - `pandas`: For data manipulation and analysis.
    - `numpy`: For numerical operations.
    - `scikit-learn`: For machine learning modeling and preprocessing.
    - `matplotlib` & `seaborn`: For data visualization.

---

## Setup and Installation

To get this project running on your local machine, follow these steps:

**1. Clone the Repository**
```sh
git clone https://github.com/ADSINGH2004/house-price-prediction.git
cd house-price-prediction
```

**2. Create and Activate a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.

*   **For macOS/Linux:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
*   **For Windows:**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    ```

**3. Install Dependencies**
Install all the required libraries from the `requirements.txt` file.
```sh
pip install -r requirements.txt
```

**4. Download the Dataset**
Download the Ames Housing Dataset (`train.csv`) from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place it inside the `data/` directory.

---

## How to Run

**1. Train the Model**
To train the model and save it as a `.pkl` file, run the training script:
```sh
python src/train.py
```
This will preprocess the data, train the Linear Regression model, and save the final model artifact to `src/model.pkl`.

**2. Make a Prediction**
You can use the `predict.py` script to load the trained model and make a prediction on new data.
```sh
python src/predict.py
```
This script will output the predicted price for a sample house defined within the file.

---

## Project Structure

```
house-price-prediction/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   └── train.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
└── src/
    ├── train.py
    ├── predict.py
    └── model.pkl
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Author

- **ADSINGH2004**
- **Date:** 2025-06-25