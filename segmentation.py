import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Display the first 5 rows
print(df.head())

# Display dataset information
print(df.info())
# Check the number of rows and columns
print("Dataset Shape:", df.shape)

# Check column names
print("Column Names:", df.columns)

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Get statistical summary
print("Statistical Summary:")
print(df.describe())
# Select features for customer segmentation
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("Selected Features:")
print(X.head())
# Find the optimal number of clusters using the Elbow Method

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# Apply K-Means Clustering

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Predict customer clusters
df['Cluster'] = kmeans.fit_predict(X)

# Display customer segments
print("Customer Segmentation Results:")
print(df.head(10))

# Display number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(df['Cluster'].value_counts())
# Visualize Customer Segments

plt.figure(figsize=(10, 6))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='rainbow',
    s=60
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
# Save customer segmentation results

df.to_csv("Customer_Segments.csv", index=False)

print("Customer segmentation results saved successfully!")
# Cluster Analysis
cluster_analysis = df.groupby("Cluster").agg({
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "Age": "mean",
    "CustomerID": "count"
})

cluster_analysis.rename(columns={
    "CustomerID": "Number of Customers"
}, inplace=True)

print("\nCustomer Segment Analysis:")
print(cluster_analysis.round(2))