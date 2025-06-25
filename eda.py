import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('AmesHousing.csv')

# --- 1. Analyze the Target Variable (SalePrice) ---

# Set up the figure and axes
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')

# Box Plot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['SalePrice'])
plt.title('Box Plot of SalePrice')
plt.xlabel('Sale Price')

plt.tight_layout()
plt.savefig('saleprice_distribution.png')
plt.clf()

# Calculate skewness and kurtosis
saleprice_skewness = df['SalePrice'].skew()
saleprice_kurtosis = df['SalePrice'].kurt()
print(f"SalePrice Skewness: {saleprice_skewness}")
print(f"SalePrice Kurtosis: {saleprice_kurtosis}\n")


# --- 2. Analyze Key Numerical Features vs. Target Variable ---

# Scatter plot: Gr Liv Area vs. SalePrice
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Gr Liv Area'], y=df['SalePrice'])
plt.title('SalePrice vs. Above Ground Living Area (Gr Liv Area)')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.savefig('saleprice_vs_grlivarea.png')
plt.clf()

# Scatter plot: Total Bsmt SF vs. SalePrice
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Total Bsmt SF'], y=df['SalePrice'])
plt.title('SalePrice vs. Total Basement Area (Total Bsmt SF)')
plt.xlabel('Total Basement Area (sq ft)')
plt.ylabel('Sale Price')
plt.savefig('saleprice_vs_totalbsmtsf.png')
plt.clf()


# --- 3. Analyze Key Categorical/Ordinal Features vs. Target Variable ---

# Box plot: Overall Qual vs. SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Overall Qual'], y=df['SalePrice'])
plt.title('SalePrice vs. Overall Quality')
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price')
plt.savefig('saleprice_vs_overallqual.png')
plt.clf()

# Box plot: Garage Cars vs. SalePrice
plt.figure(figsize=(10, 6))
# Need to handle potential NaN values for plotting
df_cleaned = df.dropna(subset=['Garage Cars'])
sns.boxplot(x=df_cleaned['Garage Cars'], y=df_cleaned['SalePrice'])
plt.title('SalePrice vs. Garage Cars')
plt.xlabel('Size of Garage in Car Capacity')
plt.ylabel('Sale Price')
plt.savefig('saleprice_vs_garagecars.png')
plt.clf()


# --- 4. Correlation Analysis ---
# Select top numerical features for correlation heatmap
# We'll include the user-specified features and a few others that are commonly important
correlation_cols = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Full Bath', 'Year Built', 'Mas Vnr Area', 'Fireplaces', 'Lot Frontage', 'Lot Area']
# Drop rows with NaN in these specific columns for a cleaner correlation matrix
df_corr = df[correlation_cols].dropna()
correlation_matrix = df_corr.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Key Features')
plt.savefig('correlation_heatmap.png')
plt.clf()

print("Detailed EDA plots have been generated.")
