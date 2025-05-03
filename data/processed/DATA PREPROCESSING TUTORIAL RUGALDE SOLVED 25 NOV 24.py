# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:37:10 2024

DATA PREPROCESSING PROJECT TUTORIAL

25 NOV 24


@author: rober ugalde
"""


import matplotlib.pyplot as plt
import seaborn as sns

# Display dataset overview
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:\n", df.describe())

# Data type of each column
print("\nData Types:\n", df.dtypes)

# Drop irrelevant columns (if needed)
columns_to_drop = ['id', 'host_id', 'name', 'host_name']  # Example
df = df.drop(columns=columns_to_drop, axis=1)

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Analyze price distribution
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Room type distribution
sns.countplot(data=df, x='room_type')
plt.title("Room Type Distribution")
plt.show()

# Scatter plot: Price vs. Number of Reviews
sns.scatterplot(data=df, x='number_of_reviews', y='price')
plt.title("Price vs Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Price")
plt.show()


# Fill missing values for columns like 'reviews_per_month'
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Drop rows with missing values
df = df.dropna()


# Example: Removing properties with prices > $1000
df = df[df['price'] <= 1000]


from sklearn.model_selection import train_test_split

# Split the data into train and test sets (80-20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Train Set Shape:", train_df.shape)
print("Test Set Shape:", test_df.shape)


# Create processed data folder
os.makedirs('./data/processed', exist_ok=True)

# Save train and test datasets
train_df.to_csv('./data/processed/train.csv', index=False)
test_df.to_csv('./data/processed/test.csv', index=False)

print("Processed datasets saved!")

xxxxxx


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
# Create folder structure if not existing
os.makedirs('./data/raw', exist_ok=True)
os.makedirs('./data/processed', exist_ok=True)

# Dataset URL
url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"

# Download and save the dataset
raw_data_path = './data/raw/AB_NYC_2019.csv'
df = pd.read_csv(url)
df.to_csv(raw_data_path, index=False)
print(f"Dataset successfully saved to {raw_data_path}")

# Load the dataset for further analysis
print("Data loaded successfully!")
print(df.head())

# Step 2: Perform EDA
# Display dataset overview
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:\n", df.describe())

# Data type of each column
print("\nData Types:\n", df.dtypes)

# Drop irrelevant columns
columns_to_drop = ['id', 'host_id', 'name', 'host_name']  # Example columns
df = df.drop(columns=columns_to_drop, axis=1)

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Analyze price distribution
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Room type distribution
sns.countplot(data=df, x='room_type')
plt.title("Room Type Distribution")
plt.show()

# Scatter plot: Price vs. Number of Reviews
sns.scatterplot(data=df, x='number_of_reviews', y='price')
plt.title("Price vs Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Price")
plt.show()

# Handle missing values
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df = df.dropna()

# Remove extreme outliers
df = df[df['price'] <= 1000]  # Example: Keeping prices <= $1000

# Step 3: Split the Dataset into Train and Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Train Set Shape:", train_df.shape)
print("Test Set Shape:", test_df.shape)

# Step 4: Save Processed Data
# Save train and test datasets
train_df.to_csv('./data/processed/train.csv', index=False)
test_df.to_csv('./data/processed/test.csv', index=False)

print("Processed datasets saved!")




